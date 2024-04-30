import numpy as np
import pandas as pd
from scipy import stats
import transform_encode as te
from multiprocessing import Manager, Pool, Value

from place_tetris import TetrisApp, trimBoard
from evals import Evals, NUM_EVALS, getEvalLabels
import os, multiprocessing

from local_params import GP, TP

FT = eval(f"lambda self, x: np.column_stack([{te.decode(TP["feature_transform"])}])")
MAX_WORKERS = TP["workers"] if TP["workers"] > 0 else multiprocessing.cpu_count()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Method to play more games if the standard deviation is not stable
# TODO: Might need to develop algo to tune theshold value
def playMore(scores, threshold=0.04, min_count=8, max_count=TP["max_plays"]):

    if len(scores) < min_count:
        return max_count  # Not enough data to make a decision

    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)
    log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
    new_aic = 2*3 - 2*log_likelihood_lognorm

    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores[:-1], floc=0)
    log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores[:-1], shape_lognorm, loc_lognorm, scale_lognorm))
    prev_aic = 2*3 - 2*log_likelihood_lognorm

    if abs(new_aic - prev_aic) / prev_aic < threshold:
        return False  # The number of games where the estimate stabilized

    return True  # Return the max games if the threshold is never met

# Method to calculate expected score
# Scores from models have high stdev and mean has heavy right skew
def expectedScore(scores, getVals=False):
    # Fit the log-normal distribution to the data
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)

    # Calculate the expected value (mean) of the log-normal distribution
    # For log-normal, mean = exp(mu + sigma^2 / 2)
    mu = np.log(scale_lognorm)  # scale_lognorm = exp(mu)
    sigma = shape_lognorm
    expected_value = np.exp(mu + (sigma**2) / 2)

    if getVals:
        return expected_value, (shape_lognorm, loc_lognorm, scale_lognorm)

    return expected_value

class Model():
    def __init__(self, tp=TP, weights=[], sigmas=np.random.uniform(0.01, 0.99, NUM_EVALS), gen=1, parent_gen=0, parentID=None, id=None):
        self.sigma = sigmas
        ft = eval(f"lambda self, x: np.array([{te.decode(tp["feature_transform"])}])")
        self.fts = ft(self, np.ones(NUM_EVALS)).shape[0]

        if len(weights) == 0 :
            weights = np.random.uniform(-15, 15, [NUM_EVALS, self.fts])

        weights = np.array(weights, copy=True)
        zero_indices = np.where(~weights.all(axis=0))[0]
        weights[:, zero_indices] = 0.01 * weights[:, 0].reshape(-1, 1)

        self.weights = weights
        self.gen = gen
        self.tp = tp
        self.parent_gen = parent_gen
        self.parentID = parentID
        if id is None:
            self.id = str(hash((gen, str(weights))))
        else:
            self.id = id
        
        self.snapshots = []

    def play(self, gp, pos, tp=TP, ft=FT, use_snap=True, it=0):

        # Loop to initialize game until a valid starting state is found if starting from snap
        while True:
            if np.random.rand() < tp['use_snap_prob'] and len(self.snapshots) > 0 and use_snap:

                # Randomly pick iteration to use snapshot from
                it_set = set([snap[0] for snap in self.snapshots])
                it_choice = np.random.choice(list(it_set))

                # Get all snapshots from the chosen iteration
                it_snaps = [snap for snap in self.snapshots if snap[0] == it_choice]

                # Randomly pick snapshot from the chosen iteration
                choice = np.random.choice(np.arange(len(it_snaps)))
                snapshot = it_snaps[choice][1]

                # Remove snapshot from list
                self.snapshots.pop(choice)

                snapscore = snapshot[1]
                game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos, snap=snapshot)

                snap_log = open(f"{CURRENT_DIR}/snap.log", "a")
                snap_log.write(f"Model ID: {self.id}\n")
                snap_log.write(f"Snapshot Used\n")
                snap_log.write(f"Score: {snapshot[1]}\n")
                snap_log.write(f"{trimBoard(snapshot[0])}\n")
                snap_log.close()

            else:
                snapscore = 0
                snapshot = None
                game = TetrisApp(gui=gp["gui"], cell_size=gp["cell_size"], cols=gp["cols"], rows=gp["rows"], sleep=gp["sleep"], window_pos=pos)
            
            if gp["gui"]:
                game.update_board()

            options = game.getFinalStates()

            if len(options) > 0:
                break

        gameover = False
        score = 0
        tot_cost = 0
        moves = 0
        norm_c_grad = np.zeros([NUM_EVALS, self.fts])
        norm_s_grad = np.zeros(NUM_EVALS)

        eval_labels = getEvalLabels()
        #print(" ".join(eval_labels)) # TODO: Comment out

        while not gameover and len(options) > 0 and (tp['demo'] or moves < tp['cutoff']): # Ends the game after CUTOFF moves unless demo
            min_cost = np.inf
            best_option = None

            for option in options:

                if option is None:
                    raise ValueError("Option is None")

                c, w_grad, s_grad, mxh = self.cost(option, tp, ft)
                if c < min_cost:
                    min_cost = c
                    best_option = option
                    min_w_grad = w_grad
                    min_s_grad = s_grad

            #print(" ".join(f"{val:.5f}" for val in w_grad[:,0]))  # TODO: Comment out
                    
            tot_cost += min_cost
            norm_c_grad += min_w_grad

            if min_s_grad is None:
                norm_s_grad = None
            else:
                norm_s_grad += min_s_grad

            moves += 1
            options, game_over, score, snapshot = game.ai_command(best_option, cp=(self, tp, ft))

            if score == 0:
                raise ValueError("Score is 0")

            # Save snapshot of the game state
            if np.random.rand() < tp['snap_prob'](mxh):
                self.snapshots.append((it, snapshot))

                snap_log = open(f"{CURRENT_DIR}/snap.log", "a")
                snap_log.write(f"Snapshot Taken\n")
                snap_log.write(f"Model ID: {self.id}\n")
                snap_log.write(f"Score: {snapshot[1]}\n")
                snap_log.write(f"{trimBoard(snapshot[0])}\n")
                snap_log.close()

        # In the event that no more points are scored
        if score - snapscore == 0:
            return score, np.zeros([NUM_EVALS, self.fts]), np.zeros(NUM_EVALS), moves

        # Return the absolute value of the average cost per move and the average gradient
        w_cost_metrics = norm_c_grad/moves/(score - snapscore) 

        if norm_s_grad is None:
            s_cost_metrics = None
        else:
            s_cost_metrics = norm_s_grad/moves/(score - snapscore)

        if moves == tp['cutoff']:
            success_log = open(f"{CURRENT_DIR}/success.log", "a")
            success_log.write(f"Game ended after {tp['cutoff']} moves\n")
            success_log.write(f"{self.weights}\n")
            success_log.close()

        game.quit_game()

        return score, w_cost_metrics, s_cost_metrics, moves
    
    def gauss(self, x, mu=0.5):
        #print(f"sigma: {self.sigma}")
        #selfprint(f"x: {x}")
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * self.sigma**2))

    def sigma_grad(self, x, mu=0.5):
        prefactor = -1 / (self.sigma**2) + ((x - mu)**2) / (self.sigma**4)
        gaussian = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((x - mu) / self.sigma)**2)
        return prefactor * gaussian

    def cost(self, state, tp=TP, ft=FT):
        vals = self.evals.getEvals(state)
        X = ft(self, vals)
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        costs = X * self.weights

        sigma_grad = None
        if "self.gauss" in tp["feature_transform"]:
            ft_start = tp["feature_transform"].index("self.gauss")
            index = np.sum([1 for char in tp["feature_transform"][:ft_start] if char == ','])
            gWeights = self.weights[:, index]
            sigma_grad = self.sigma_grad(vals) * gWeights

        mxh = vals[1] # Save max height for snapshot

        return np.sum(costs), X, sigma_grad, mxh

    def evaluate(self, args):
        id, plays, gp = args

        self.evals = Evals(gp)

        # Won't always put window in same place bc proccesses will finish in unknown order
        slot = id % MAX_WORKERS

        width = gp["cell_size"] * (gp["cols"] + 6)
        height = gp["cell_size"] * gp["rows"] + 80
        pos = ((width * slot) % 2560, height * int(slot / int(2560 / width)))

        scores = np.zeros(plays)
        shape = self.weights.shape
        # 1 for score, rest for cost metrics not including bias term
        w_cost_metrics_lst = np.zeros((plays,shape[0],shape[1])) 
        s_cost_metrics_lst = np.zeros((plays, shape[0]))
        for i in range(plays):
            score, w_cost_metrics, s_cost_metrics, _ = self.play(gp, pos, TP, it=i)
            scores[i] = score
            w_cost_metrics_lst[i] = w_cost_metrics
            s_cost_metrics_lst[i] = s_cost_metrics

            if not playMore(scores[:i+1]):
                break

        # Trim to only played games before calculating expected score
        scores = scores[:i+1]
        score, ln_vals = expectedScore(scores, True)

        # Improve weighting of cost gradient of poorly preforming plays by preforming a log-norm transform on the metrics 
        # based on how they are distributed among the play throughs and summing the results of the transform
        w = stats.lognorm.pdf(scores, *ln_vals)

        w_cost_metrics = np.sum(w_cost_metrics_lst[:i+1] * w[:, np.newaxis, np.newaxis], axis=0)
        s_cost_metrics = np.sum(s_cost_metrics_lst[:i+1] * w[:, np.newaxis], axis=0)
        
        std = np.std(scores)

        shape, _, scale = ln_vals

        def rank(shape, scale):
            exp = np.log(scale)
            std = shape
            expected_value = np.exp(exp - std/2)
    
            return expected_value

        def aic(scores):
            shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(scores, floc=0)
            log_likelihood_lognorm = np.sum(stats.lognorm.logpdf(scores, shape_lognorm, loc_lognorm, scale_lognorm))
            return 2*3 - 2*log_likelihood_lognorm

        df = pd.DataFrame({
            'gen': [self.gen],
            'rank': [rank(shape, scale)],
            'exp_score': [score],
            'std': [std],
            'aic': [aic(scores)],
            'shape': [shape],
            'scale': [scale],
            'model': [self],
            'id': [self.id],
            'parentID': [self.parentID],
            'weights': [self.weights.flatten()],
            'sigmas': [self.sigma],
            'w_cost_metrics': [w_cost_metrics.flatten()],
            's_cost_metrics': [s_cost_metrics],
        })

        return df

    def mutate(self, gen, w_grad=None, s_cm=None, max_children=-1):

        # TODO: Introduce momentum factor

        if w_grad is None:
            w_grad = np.ones([NUM_EVALS, self.fts])

        if s_cm is None:
            s_cm = np.ones(NUM_EVALS)

        age_factor = min(100, TP['age_factor'](gen - self.gen))

        # Regularize gradient step scale largest step to the learning rate
        no_step = np.where(np.all(w_grad == 1, axis=0)) # Don't step on columns of grad that are all init as 1 (used when creating new FT)
        w_l2 = np.linalg.norm(self.weights, axis=0)
        w_step = TP["learning_rate"](gen) * w_l2 * w_grad/np.linalg.norm(w_grad, axis=0)
        w_step[:, no_step] = 0

        s_l2 = np.linalg.norm(self.sigma)
        s_step = TP["s_learning_rate"](gen) * s_l2 * s_cm/np.linalg.norm(s_cm)
        
        if max_children == -1: # For lineage limiting
            max_children = TP["population_size"]*TP["lin_prop_max"]
        nchildren = min(int(TP["population_size"] * (1 - TP["p_random"]) / TP["top_n"]), int(max_children))
        children = []

        # Regularize mutation
        w_strength = TP["mutation_strength"](gen) * w_l2
        s_strength = TP["s_mutation_strength"](gen) * s_l2
        for _ in range(nchildren):
            new_weights = self.weights.copy()

            # Age factor for learning rate (randomly increases or decreases learning rate based on age factor)
            lr_age_factor = age_factor if np.random.rand() < 0.5 else 1 / age_factor
            w_step_af = w_step * lr_age_factor

            # Check if bias term is included in feature transform
            index = None
            if "np.ones_like(x)" in self.tp["feature_transform"]:
                bias_start = self.tp["feature_transform"].index("np.ones_like(x)")
                index = np.sum([1 for char in self.tp["feature_transform"][:bias_start] if char == ','])

                # Excludes bias term from stepping gradient
                new_weights[:, np.arange(new_weights.shape[1]) != index] -= w_step_af[:, np.arange(new_weights.shape[1]) != index]

            else:
                new_weights -= w_step_af

            new_sigmas = self.sigma.copy()
            new_sigmas -= s_step

            # Random mutation introduction
            for e in range(NUM_EVALS):
                for f in range(self.fts):
                    if np.random.rand() < TP["mutation_rate"](gen):
                        # Strong mutation if step is 0
                        if w_step[e,f] == 0:
                            strength = w_strength[f] * 500 # Strengthern mutation strength on newly minted FT
                        else:
                            strength = w_strength[f]

                        # Strengthen mutation for bias
                        if (index is not None) and (f == index):
                            strength *= 5

                        # Strengthen mutation for very old parents (max age factor is 100)
                        age_factor = min(100, TP['age_factor'](gen - self.gen))
                        mutation = np.random.normal(0, strength, 1)[0] # Can increase or decrease weights
                        
                        new_weights[e,f] += mutation * age_factor

            for i in range(len(new_sigmas)):
                if np.random.rand() < TP["mutation_rate"](gen):
                    new_sigmas[i] += age_factor * np.random.normal(0, s_strength, 1)[0]

            new_sigmas = np.clip(new_sigmas, 0.01, 0.99)

            model = Model(weights=new_weights, sigmas=new_sigmas, parent_gen=gen, parentID=self.id, tp=self.tp)
            model.tp = None # Remove reference to TP for pickling
            children.append(model)

        return pd.DataFrame(children, columns=['model'])
