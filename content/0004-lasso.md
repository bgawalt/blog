Title: Why's Lasso Do That?
Date: 2026-01-13 11:30
Slug: 0004-lasso
Category: stats
Tags: stats, ml, lasso
Author: Brian Gawalt
Summary: How the Lasso forces a zero-weight model: an end-to-end rundown.
opengraph_image: 0004_lasso_twirl.png


![MS Paint doodle of a purple lambda (the Greek alphabet character) twirling alasso in the desert](/images/0004_lasso_twirl.png){: style="width:80%; max-width:500px;"}

> **Note:** this post makes heavy use of MathJax. If you're reading via RSS,
> you'll want to click through to the web version.

My intuition of regularization is: it's a compromise.  You want a model that
fits your historical examples, but you also want a model that is "simple."
So you set some exchange rate -- the strength of regularization -- and trade off
"fit my historical data" against "be a simple model." You meet somewhere in the
middle: a model that's simpler than your unregularized fit would produce, but
not *so* simple to the point that it's missing obvious/robust patterns in the
training data.

In the biz, we call it a penalty on complexity, which is different than calling
it a *ban* on complexity.  We call it shrinking, which is different than calling
it *vanishing.*  These names reflect the intuition: penalize something to get
less of it, but not none of it; shrink something to make it smaller, not
to make it disappear.  With regularization, we'll reach some compromise point,
and get a model that (a) is less well-fit to the training data than in the
unregularized state, but also (b) not maximally, uselessly "simple."

This blog post is about how one of the world's two most famous regularized
regression schemes, the Lasso
([Tibshirani 1996](https://www.jstor.org/stable/2346178)), rejects compromise.
At and beyond a certain penalty rate, it will quite happily *only* give you a
maximally-simple model. No compromise, just an empty model; "I found no pattern
in the data."  And people love this about the Lasso!
[My dissertation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-252.html)
was built on this property, where Lasso regularization can zero out model
weights.

But it's also weird to me that it's possible, given what I thought we were doing
by regularizing a model fit.  It's not a compromise anymore.
Why's Lasso do that?

## What's Lasso?  (A convex optimization.)

We'll formalize "fit the data, but also use a simple model" by posing an
optimization task built on five raw ingredients:

1.  Defining "the data" as a collection of $N$ vector-scalar pairs,
    $\left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N$, where each
    $\vec{\mathbb{x}}_j$ is in $\mathbb{R}^p$ (call each of these a *feature
    vector*) and each $y_j$ is a scalar *label*.
2.  Defining the model as a vector of $p$ parameters,
    $\vec{\mathbb{w}} \in \mathbb{R}^p$.  Call each individual parameter, each
    element of this vector $w_i,~i = 1, \ldots, p$, a *model weight.*
3.  Defining a function, $f\left(\cdot; \left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N\right): \mathbb{R}^p \to \mathbb{R}$,
    that expresses goodness-of-fit.  By construction of $f$, a model parameter
    vector that minimizes $f$ corresponds to a model that's "fitting the data"
    to the best possible extent.  Call $f$ the *loss function*.
4.  Defining a function, $r: \mathbb{R}^p \to \mathbb{R}$, that expresses
    model simplicity.  A model parameter vector that minimizes this function
    corresponds to a maximally simple model.  Call $r$ the *regularizer.*
5.  Defining a scalar hyperparameter, $\lambda \geq 0$, that defines the strength
    of regularization.  The bigger $\lambda$, the more we favor $r$ over $f$.

Mixing these together, our optimization task is:

$$\vec{\mathbb{w}}^* := \arg \min_{\vec{\mathbb{w}} \in \mathbb{R}^p} f\left(\vec{\mathbb{w}}; \left\{\vec{\mathbb{x}}_j, y_j\right\}\right) + \lambda r(\vec{\mathbb{w}})$$

### Lasso's loss function

Our loss function is defined as:

$$f\left(\vec{\mathbb{w}}; \left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N\right) = \sum_{j=1}^N{\left(\vec{\mathbb{w}}^T\vec{\mathbb{x}}_j - y_j\right)^2}$$ 

That's:

1. the sum, over all $N$ features-and-label pairs,
2. of the square
3. of the difference
4. between the actual label $y_j$
5. and the dot product of $\vec{\mathbb{w}}$ and $\vec{\mathbb{x}}_j$

The model weights "fit the data" by dint of defining a weighted sum of the
feature vector elements that is a reliably close match to the features'
corresponding scalar label.  The loss function is a sum of squares, so it's
always non-negative; the closer it is to zero, the better the average label 
match we are getting from our model weights.  Minimize the loss function to
best fit the data, just how we wanted.

### Lasso's regularizer

Our regularizer is:

$$r\left(\vec{\mathbb{w}}\right) = \left|\left|\vec{\mathbb{w}}\right|\right|_1 = \sum_{i=1}^p\left|w_i\right|$$

the $L_1$ norm of our model weight vector.  Models with weights that are
reliably close to zero count as "simpler" than ones with larger-magnitude 
weights.  Just like $f$, $r$ is always non-negative, and we can minimize it by
choosing a weight vector of all zeroes.  The model that always picks "zero" as
its guess for $y$, totally ignoring the feature vector, is this regularizer's
simplest possible model.

### Lasso's convexity

From above, our optimization task is:

$$\vec{\mathbb{w}}^* := \arg \min_{\vec{\mathbb{w}} \in \mathbb{R}^p} f\left(\vec{\mathbb{w}}; \left\{\vec{\mathbb{x}}_j, y_j\right\}\right) + \lambda r(\vec{\mathbb{w}})$$

Take on faith that this mixture of $f$, $r$, and $\lambda$ is convex.
$f$ and $r$ are both convex, and $\lambda$ is non-negative, which lets us argue

1.  the scaling-up of a convex function by a non-negatve value is itself convex,
    so $\lambda r(\vec{\mathbb{w}})$ is also convex, 
2.  the sum of two convex functions is itself convex, so the overall mixture
    $(f + \lambda r)$ we're minimizing is convex.

Convex functions come with a lot of nice properties I will leverage later on, so
I want to emphasize the convexity of our optimization task now.

### Dropping down to univariate Lasso

From here, we'll take a special case: when $p$ is 1.  This univariate regression
turns our weight and feature "vectors" into plain ol' scalars, making our
optimization task:

$$w^* := \arg \min_{w \in \mathbb{R}} \sum_{j=1}^N{\left(w \cdot x_j - y_j\right)^2} + \lambda\left|w\right|$$ 

I promise we're reinflate this to the multivariate case before we're done here.
Lasso's convexity will help a lot with that.

But now that we're down here, we celebrate that low dimensionality means we can
make charts.  Charts!

Here's a dataset with a clear relationship between $x$ and $y$.  The data are
plotted as blue dots.  I fit three Lasso models to this dataset, with $\lambda$
taking on values of 0, 200, or 600.  I overlay the dashed trendlines each model
produces for the range of $x$'s, and we see those three trendlines decay from
a slope of 1 to a totally null (maximally simple!) slope of zero:

![A scatterplot of blue dots whose x values are uniformly distributed from -3 to
3.  Their y values largely track the x values, up to an additive factor of
+/- 0.5 or so.  Three dashed lines pass through the origin, colored blue,
purple, and red.  They correspond to model predictions from an unregularized
(slope of 0.99), moderately regularized (slope 0.588), and heavily regularized
(slope 0) Lasso fit.](/images/0004_scatterplot.png)

If we repeatedly produce a model weight $w*(\lambda)$ for many values of
$\lambda$, we can see this decay evolve in more detail (with our three models
above appearing as special large dots):

![The regularization path that results from sweeping lambda from 0 to 800 or so.
The y-axis is labelled w*(lambda), ranging from 0 to 1.  The graph's title is
"Regularization Path".  A single red series starts as a straight red line from 
the point (0, 0.99) to (500, 0), connecting to another straight red line from
(500, 0) to (800, 0).  Three dots appear, a blue one at (0, 0.99), purple at
(200, 0.588), and red at (600, 0).](/images/0004_regularization_path.png)

As regularization strength increases, the resulting model weight drops linearly,
until it hits absolute zero.  Early on, you see compromise between $f$ and $r$:
$f$ wants a slope of 1, and $r$ wants a slope of 0, and as we strengthen $r$'s
negotiating leverage, we get models that look more and more like what $r$ wants.

But past $\lambda = 500$ or so, the optimization is ignoring the $f$ completely.
That fit-the-data component is still there as part of the objective to be
minimized, and yet there's no compromise: the model weight returns what $r$
wants, and only what $r$ wants.  It's *like* $f$ isn't even there, except it
*is.*

What's Lasso doing?  Why's Lasso do that?


## What's Lasso doing?  (Parabolas.)

Lasso is making our optimizer follow the instructions of one of two parabolas.
When we strengthen regularization past a certain point, we're not able to make
either parabola happy, and bounce back and forth between them until we're stuck
at $w^* = 0$.

### Zero regularization

When there's no regularization at all, when $\lambda = 0$, our optimizer's
objective is just to minimize the loss function $f$, the sum of squared errors:

$$\begin{align}w^*(0) &= \arg \min_{w \in \mathbb{R}^p} f\left(w; \left\{x_j, y_j\right\}\right)\\
  &= \arg \min_{w \in \mathbb{R}} \sum_{j=1}^N{\left(wx_j - y_j\right)^2}\end{align}$$

If we rearrange $f$'s terms a bit, we'll see it's a parabola as a function of
$w$:

$$\begin{align}f\left(w; \left\{x_j, y_j\right\}\right) &= \sum_{j=1}^N{\left(wx_j - y_j\right)^2}\\
  &= \sum_{j=1}^N \left[x_j^2 w^2 - 2 x_j y_j w + y_j^2\right] \\
  &= \left[\sum_{j=1}^N x_j^2\right] w^2 - 2\left[\sum_{j=1}^N (x_j y_j)\right] w + \left[\sum_{j=1}^N y_j^2\right]\end{align}$$

Define some helpful aliases for those terms,

$$S_x := \sum_{j=1}^N x_j^2~~~~D_{xy} := \sum_{j=1}^N (x_j y_j)~~~~S_y := \sum_{j=1}^N y_j^2$$

where $S$'s stand for sums of squares our features or labels and $D$ stands for
a dot product of feature and label.  That makes the parabolaness of the
unregularized loss function really stand out:

$$f\left(w; \left\{x_j, y_j\right\}\right) = S_xw^2 - 2D_{xy}w + S_y$$

The vertex of a parabola like this lies at the famous "$-b/2a$" point, which
means $f$ is minimized at:

$$w^*(0) = -(-2D_{xy})/(2S_x) = D/S_x$$

(Or, expanding those terms for the fun of it:)

$$w^*(0) = \frac{\sum_{j=1}^N (x_j y_j)}{\sum_{j=1}^N x_j^2}$$

Let's plot it for the same blue-dots dataset we used above:

![A blue convex parabola, crossing the y axis at around 340, hitting its minimum
at the point (1.03, 13.9), with its axis of symmetry added as a vertical blue
dashed line](/images/0004_parabola_zero_reg.png)

Imagine our optimizer as a little guy, wandering back and forth along the range
of $w$ to minimize $f$.  This parabola gives our optimizer unambiguous
instructions:

> Wherever you are right now, move towards $(D_{xy}/S_x)$.  As long as each step
> puts you lower than you were before, you will find the minimum.

### Some regularization

When we increase $\lambda$ to some positive value, are adding $\lambda r(w)$ to
our loss function parabola.  That scaled regularizer is a function that looks
like:

$$\begin{align}\lambda r(w) &= \lambda\left|w\right| \\
  &= \left\{
        \begin{array}{rl} -\lambda w;&w < 0~~\mbox{(LHS)} \\
          \lambda w;&w \geq 0~~\mbox{(RHS)}
        \end{array}
    \right.\end{align}$$

It's piecewise linear function, with a negative slope on the left-hand side
(LHS) and positive slope on the right-hand side (RHS).  When we add $f$ to this,
we wind up with a piecewise quadratic function:

$$\begin{align}f(w) + \lambda r(w) &= f(w) + \lambda\left|w\right| \\
    &= \left\{
        \begin{array}{rl} f(w) -\lambda w;&w < 0~~\mbox{(LHS)} \\
          f(w) + \lambda w;&w \geq 0~~\mbox{(RHS)}
        \end{array}
    \right. \\
    &= \left\{
        \begin{array}{rl} S_xw^2 - 2D_{xy}w + S_y -\lambda w;&w < 0~~\mbox{(LHS)} \\
          S_xw^2 - 2D_{xy}w + S_y + \lambda w;&w \geq 0~~\mbox{(RHS)}
        \end{array}
    \right. \\
    &= \left\{
        \begin{array}{rl} S_xw^2 - (2D_{xy} + \lambda)w + S_y;&w < 0~~\mbox{(LHS)} \\
          S_xw^2 - (2D_{xy} - \lambda) w + S_y;&w \geq 0~~\mbox{(RHS)}
        \end{array}
    \right.
\end{align}$$

These two parabolas, LHS and RHS, have "$-b/2a$" vertices aligned at:

$$\mbox{LHS vertex:}~~w_{LHS}(\lambda) = \frac{2D_{xy} + \lambda}{S_x}$$
$$\mbox{RHS vertex:}~~w_{RHS}(\lambda) = \frac{2D_{xy} - \lambda}{S_x}$$

*This* function provides our little-guy optimizer a more complicated set of
instructions:

> *  **While you are on the LHS:** take descent steps towards
>    $(2D_{xy} + \lambda)/S_x$.  If your latest step took you out of the LHS,
>    this rule no longer applies.
> *  **While you are on the RHS:** take descent steps towards
>    $(2D_{xy} - \lambda)/S_x$.  If your latest step took you out of the RHS,
>    this rule no longer applies.

When regularization is non-zero, but *light,* the optimizer can resolve this
without too much extra effort.  Here's what the mixed objective looks like
for our blue-dots dataset (the purple curve), where $f$'s vertex is on the RHS:

![Our blue parabola from "zero regularization" now joined by a red scaled 
absolute value function that passes through (1, 200).  Both these are mostly
transparent, they are called "f(w)" and "lambda r(w)" in the legend at the 
bottom of the figure.  Their sum, a thicker, opaque purple series, looks an
awful lot like a parabola.  Its vertex at (QQQ, QQQ) is marked with a large
dot.](/images/0004_parabola_some_reg.png)

Imagine the little guy starts at $w = -1$.  His thought process is:

1.  "I am starting on the LHS.  The LHS instructions say I should head to
    $w = (2D_{xy} + \lambda)/S_x$."
2.  **\[later:\]** "I've taken a lot of steps, and not reached
    $(2D_{xy} + \lambda)/S_x$, and I have in fact crossed the $w = 0$ border.
    I'm on the RHS now.  My instructions from Step 1 no longer apply."
3.  "My new instructions, now that I'm on the RHS, say to head to
    $w = (2D_{xy} - \lambda)/S_x$ instead."
4.  **\[later:\]** "OK, made it.  I am at the vertex of the RHS parabola, which
    is on the RHS (i.e., I never re-crossed the $w = 0$ border)."
5.  "My instructions are still the same as they were in Step 3, and I've
    finished following them.
    [Job's done.](https://www.youtube.com/watch?v=5r06heQ5HsI)"

TODO: redraw parabolas but with shrinkage annotated

### Lots of regularization

![Our blue parabola from "zero regularization" joined by a steeper red scaled 
absolute value function that passes through (0.5, 350).  Both these are mostly
transparent, they are called "f(w)" and "lambda r(w)" in the legend at the 
bottom of the figure.  Their sum, a thicker, opaque red series, no longer looks
like a parabola, it now looks like a Greek nu character with it point at
(0, 320).](/images/0004_parabola_lots_reg.png)

## Why's Lasso do that? (Discontinuous slope.)

## Returning to multiple features
