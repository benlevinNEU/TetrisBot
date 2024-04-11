## Cost Function

$$
C = \sum_{g=1}^{G} pdf(S_g) * C_g
$$

Where:

$$
pdf(S_g) = \frac{1}{S_g \sigma\sqrt{2\pi}} * exp\left(-\frac{(ln(S_g)-\mu)^2}{2\sigma^2}\right)  \quad \quad
C_g = \frac{1}{N_gS_g} \sum_{m=1}^{N_g} C_{g,m}
$$

$$
\mu = \frac{1}{G} \sum_{g=1}^{G} ln(S_g) \quad \quad

\sigma = \sqrt{\frac{1}{G} \sum_{g=1}^{G} (ln(S_g)-\mu)^2}
$$

Where:

$$
C_{g,m} = \sum_{e=1}^{E} w_e \cdot f(x_{g,m,e})^T = W \cdot F(X_{g,m})^T
$$

Where:

$$
W = \begin{bmatrix} w_{1,1} & w_{1,2} & \cdots & w_{1,FS} \\ w_{2,1} & w_{2,2} & \cdots & w_{2,FS} \\ \vdots & \vdots & \ddots & \vdots \\ w_{E,1} & w_{E,2} & \cdots & w_{E,FS} \end{bmatrix}

\text{ and } 

X_{g,m} = \begin{bmatrix} x_{g,m,1} \\ x_{g,m,2} \\ \vdots \\ x_{g,m,E} \end{bmatrix}
$$

Or, in other words:

$$
C = \sum_{g=1}^{G}\frac{1}{S_g \left( \sqrt{\frac{1}{G} \sum_{g=1}^{G} \left(ln(S_g)-\frac{1}{G} \sum_{g=1}^{G} ln(S_g)\right)^2} \right)\sqrt{2\pi}} * exp\left(-\frac{\left(ln(S_g)-\left( \frac{1}{G} \sum_{g=1}^{G} ln(S_g) \right)\right)^2}{ \frac{2}{G} \sum_{g=1}^{G} \left(ln(S_g)-\frac{1}{G} \sum_{g=1}^{G} ln(S_g)\right)^2 }\right) * \frac{1}{N_gS_g} \sum_{m=1}^{N_g} \sum_{e=1}^{E} w_e \cdot f(x_{g,m,e})^T
$$

Where:
- \( $G$ \) is the Number of games played by the model.
- \( $S_g$ \) is the Final score of the \( $g^{th}$ \) game.
- \( $ pdf(S_g) $ \) is the Probability density function of the score of the \( $g^{th}$ \) game.
- \( $C_g$ \) is the Total cost of the \( $g^{th}$ \) game.
- \( $N_g$ \) is the Number of moves in the \( $g^{th}$ \) game.
- \( $E$ \) is the Number of evaluation metrics.
- \( $FS$ \) is the Number of feature transforms.
- \( $w_e$ \) is the Weight of the \( $e^{th}$ \) feature transform.
- \( $f(x)$ \) is the Feature transformation function.
- \( $x_{g,m,e}$ \) is the Raw evaluation metric for game $g$, move $m$ and metric $e$.

In this function:
- Each move's cost is determined by applying a feature transformation function \( $f$ \) to each evaluation metric \( $x_{g,m,e}$ \), then scaling by a corresponding weight array \( $w_e$ \) that matches the size of the feature transform.
- The total cost \( $C$ \) is the sum of these scaled values over all moves in a game, normalized by the number of moves \( $N_g$ \) and the final score \( $S_g$ \) of game $g$ then averaged overall all games played by the model.

### Partial Derivatives for Cost Function

$$
\frac{\partial C}{\partial w_i} = \frac{1}{G}\sum_{g=1}^{G}\frac{1}{N_gS_g} \sum_{m=1}^{N_g}\sum_{e=1}^{E} f_i(x_{g,m,e})
$$

$$
\frac{\partial C_{gauss}}{\partial \sigma} = \frac{1}{G}\sum_{g=1}^{G}\frac{1}{N_gS_g} \sum_{m=1}^{N_g}\sum_{e=1}^{E} w_{e, gauss} \left[ -\frac{1}{\sigma^2} + \frac{(x_{g,m,e}-\mu)^2}{\sigma^4} \right] \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x_{g,m,e}-\mu}{\sigma}\right)^2}
$$
