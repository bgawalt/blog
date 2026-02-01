Title: Why's Lasso Do That?
Date: 2026-02-02 11:30
Slug: 0004-lasso
Category: stats
Tags: stats, ml, lasso
Author: Brian Gawalt
Summary: How the Lasso forces a zero-weight model: an end-to-end rundown.
opengraph_image: 0004_lasso_twirl.png


![MS Paint doodle of a purple lambda (the Greek alphabet character) twirling a
lasso in the desert](/images/0004_lasso_twirl.png){: style="width:80%; max-width:500px;"}

> **Note:** this post makes heavy use of MathJax. If you're reading via RSS,
> you'll want to click through to the web version.

My intuition of regularization is: it's a compromise.  You want a model that
fits your historical examples, but you also want a model that is "simple."
So you set some exchange rate -- the strength of regularization -- and trade off
"fit my historical data" against "be a simple model." You meet somewhere in the
middle: a model that's simpler than your unregularized fit would produce, but
not *so* simple to the point that it's missing obvious/reliable patterns in the
training data.

In the biz, we call it a penalty on complexity, which is different than calling
it a *ban* on complexity.  We call it shrinking, which is different than calling
it *vanishing.*  These names reflect the intuition: penalize something to get
less of it, but not none of it; shrink something to make it smaller, not
to make it disappear.  With regularization, we'll reach some compromise point,
and get a model that (a) is less well-fit to the training data than in the
unregularized state, but also (b) not maximally, uselessly "simple."

This blog post is about how the world's most regularized regression schemes,
the Lasso ([Tibshirani 1996](https://www.jstor.org/stable/2346178)), rejects
compromise. For certain, sufficiently-high penalty rates, it will quite happily
*only* give you a maximally-simple model: one that zeros out consideration of
any feature on which you tried to base your predictions. No compromise, just an
all-zeros empty model; "I found no pattern in the data."  And people love this
about the Lasso!
[My dissertation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-252.html)
was built on this property, where Lasso regularization can zero out model
weights.

But it's also weird to me that it's possible, given what I thought we were doing
by regularizing a model fit.  It's not a compromise anymore.

Why's Lasso do that? 

## What's Lasso?  (A convex optimization.)

It's typical for statisticians to formalize "fit the data, but with a simple
model" by posing an optimization task built on five raw ingredients:

1.  Defining "the data" as a collection of $N$ vector-scalar pairs,
    $\left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N$, where each
    $\vec{\mathbb{x}}_j$ is in $\mathbb{R}^p$ (call each of these a *feature
    vector*) and each $y_j$ is a scalar *label*.
2.  Defining the model as a vector of $p$ parameters,
    $\vec{\mathbb{w}} \in \mathbb{R}^p$.  Call each individual parameter, each
    element of this vector $w_i,~i = 1, \ldots, p$, a *model weight.*
3.  Defining a function,
    $f\left(~\cdot~; \left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N\right): \mathbb{R}^p \to \mathbb{R}$,
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
weights.  Just like $f$, $r$ is always non-negative.  We can minimize it by
choosing a weight vector of *all zeroes.*  The model that always picks "zero" as
its guess for $y$, totally ignoring the feature vector, is this regularizer's
simplest possible model.

### Lasso's convexity

Altogether, the optimization task is:

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
taking on values of 0, 350, or 800.  I overlay the dashed trendlines each model
produces for the range of $x$'s, and we see those three trendlines decay from
a slope of 1 to a totally null (maximally simple!) slope of zero:

![A scatterplot of blue dots whose x values are uniformly distributed from -3 to
3.  Their y values largely track the x values, up to an additive factor of
+/- 0.5 or so.  Three dashed lines pass through the origin, colored blue,
purple, and red.  They correspond to model predictions from an unregularized
(slope of 1.037), moderately regularized (slope 0.461), and heavily regularized
(slope 0) Lasso fit.](/images/0004_a_scatterplot.png)

If we repeatedly produce a model weight $w^*(\lambda)$ for many values of
$\lambda$, we can see this decay evolve in more detail (with our three models
above appearing as special large dots):

![The regularization path that results from sweeping lambda from 0 to 800 or so.
The y-axis is labelled w*(lambda), ranging from 0 to just over 1.  The graph's
title is "Regularization Path".  A single red series starts as a straight red
line from  the point (0, 1.037) to (620, 0), connecting to another straight red
line from (620, 0) to (950, 0).  Three dots appear, a blue one at (0, 1.037),
purple at (350, 0.461), and red at (800, 0).](/images/0004_b_regularization_path.png)

As regularization strength increases, the resulting model weight drops linearly,
until it hits absolute zero.  Early on, you see compromise between $f$ and $r$:
$f$ wants a slope of 1, and $r$ wants a slope of 0, and as we strengthen $r$'s
negotiating leverage, we get models that look more and more like what $r$ wants.

But past $\lambda = 620$ or so, the optimization is ignoring the $f$ completely.
That fit-the-data component is still there as part of the objective to be
minimized, and yet there's no compromise: the model weight returns what $r$
wants, and only what $r$ wants.  It's *like* $f$ isn't even there, except it
*is.*  We never zero'ed $f$, we just amp'ed up $r$.

What's Lasso doing?  Why's Lasso do that?


## What's Lasso doing?  (Parabolas.)

Lasso is making our optimizer follow the instructions of a pair of parabolas.
When we strengthen regularization past a certain point, we're not able to make
either parabola happy, and bounce back and forth between them until we're stuck
at $w^* = 0$.

### Zero regularization

When there's no regularization at all, at $\lambda = 0$, our optimizer's
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
at the point (1.038, 13.9), with its axis of symmetry added as a vertical blue
dashed line](/images/0004_c_parabola_zero_reg.png)

Imagine our optimizer as a little guy, wandering back and forth along the range
of $w$ to minimize $f$.  This parabola gives our little optimizer guy
unambiguous instructions:

> Wherever you are right now, move towards $(D_{xy}/S_x)$.  As long as each step
> puts you lower than you were before, you will find the minimum.

If the little optimizer guy loops that enough, he'll land at $w^*(0)$.

### Some regularization

When we increase $\lambda$ to some positive value, are adding $\lambda r(w)$ to
our loss function parabola.  That scaled regularizer is a function that looks
like:

$$\begin{align}\lambda r(w) &= \lambda\left|w\right| \\
  &= \begin{cases}
        -\lambda w &w < 0~~\mbox{(LHS)} \\
        \lambda w &w \geq 0~~\mbox{(RHS)}
        \end{cases}
\end{align}$$

It's a piecewise linear function, with a negative slope on the left-hand side
(LHS) and positive slope on the right-hand side (RHS).  When we add $f$ to this,
we wind up with a piecewise quadratic function:

$$\begin{align}f(w) + \lambda r(w) &= f(w) + \lambda\left|w\right| \\
    &= \begin{cases}
          f(w) -\lambda w &w < 0~~\mbox{(LHS)} \\
          f(w) + \lambda w &w \geq 0~~\mbox{(RHS)}
    \end{cases} \\
    &= \begin{cases}
          S_xw^2 - 2D_{xy}w + S_y -\lambda w&w < 0~~\mbox{(LHS)} \\
          S_xw^2 - 2D_{xy}w + S_y + \lambda w&w \geq 0~~\mbox{(RHS)}
    \end{cases} \\
    &= \begin{cases}
          S_xw^2 - (2D_{xy} + \lambda)w + S_y &w < 0~~\mbox{(LHS)} \\
          S_xw^2 - (2D_{xy} - \lambda) w + S_y &w \geq 0~~\mbox{(RHS)}
    \end{cases}
\end{align}$$

These two parabolas, LHS and RHS, have "$-b/2a$" vertices aligned at:

$$\mbox{LHS vertex:}~~w_{LHS}(\lambda) = \frac{2D_{xy} + \lambda}{2S_x}$$
$$\mbox{RHS vertex:}~~w_{RHS}(\lambda) = \frac{2D_{xy} - \lambda}{2S_x}$$

*This* function provides our little-guy optimizer a more complicated set of
instructions:

> *  **While you are on the LHS:** take descent steps towards
>    $(2D_{xy} + \lambda)/(2S_x)$.  If your latest step took you out of the LHS,
>    this rule no longer applies.
> *  **While you are on the RHS:** take descent steps towards
>    $(2D_{xy} - \lambda)/(2S_x)$.  If your latest step took you out of the RHS,
>    this rule no longer applies.

When regularization is non-zero, but *light,* the optimizer can resolve this
without too much extra effort.  We saw a case like that above, when we set
$\lambda := 350$ for our blue-dots dataset. Here's what the mixed objective
looks like:

![Our blue parabola from "zero regularization" now joined by a red scaled 
absolute value function that passes through (1, 350).  Both these are mostly
transparent, they are called "f(w)" and "lambda r(w)" in the legend at the 
bottom of the figure.  Their sum, a thicker, opaque purple series, looks an
awful lot like a parabola.](/images/0004_d_parabola_some_reg.png)

Imagine the little guy starts at $w = -1$.  His thought process is:

1.  "I am starting on the LHS.  The LHS instructions say I should head to
    $w = (2D_{xy} + \lambda)/(2S_x)$."
2.  **\[later:\]** "I've taken a lot of steps, and not reached
    $(2D_{xy} + \lambda)/(2S_x)$, and I have in fact crossed the $w = 0$ border.
    I'm on the RHS now.  My instructions from Step 1 no longer apply."
3.  "My new instructions, now that I'm on the RHS, say to head to
    $w = (2D_{xy} - \lambda)/(2S_x)$ instead."
4.  **\[later:\]** "OK, made it.  I am at the vertex of the RHS parabola, which
    is on the RHS (i.e., I never re-crossed the $w = 0$ border)."
5.  "My instructions are still the same as they were in Step 3, and I've
    finished following them.
    [Job's done.](https://www.youtube.com/watch?v=5r06heQ5HsI)"

The difference between $w^*(0)$ to $w^*(350)$ is *shrinkage*: the vertex of the
RHS parabola is $-\lambda/(2S_x)$ units smaller than the zero-regularization
vertex:

![Same as previous image, but with the vertices of the blue and purple
parabolas drawn with large blue/purple dots.  Two vertical dashed guidelines
mark the axes of symmetry for the parabolas, and a left-pointing arrow going
from the blue axis to the purple axis is labeled "-λ/(2Sx)"](/images/0004_e_parabola_some_reg_vertices.png)

Neither $f$ nor $r$ are minimized at that purple vertex axis of symmetry.  Both
could be much smaller, if they didn't have to accommodate the other.  That's
a good compromise: one that leaves everyone upset.

### Lots of regularization

When we ramp up our value of $\lambda$ from 350 to 800 on our blue-dots example,
nothing about the rules change for our little-guy optimizer.  What does change
is the layout of the two parabolae, such that neither parabola's vertex lies
on the side its rule governs:

![Our blue parabola from "zero regularization" joined by a steeper red scaled 
absolute value function that passes through (0.5, 350).  Both these are mostly
transparent, they are called "f(w)" and "lambda r(w)" in the legend at the 
bottom of the figure.  Their sum, a thicker, opaque red series, no longer looks
like a parabola, it now looks like a Greek nu character with it point at
(0, 320).](/images/0004_f_parabola_lots_reg.png)

The thought process for our little guy is:

1.  "When I'm on the LHS, it wants me to go to
    $\left(2D_{xy} + \lambda\right)/(2S_x)$, which is on the RHS."
2.  "But then when I get to the RHS, *it* wants me to go look for a minimum at
     $\left(2D_{xy} - \lambda\right)/(2S_x)$, which is back on the LHS.
     I just came from there!!"

For some optimizers, this bouncing back and forth can take awhile to resolve.
But when it does resolve, it can only end at $w^*(800) = 0$, the maximally
simple model weight.

When does this happen?  It happens when both the LHS and RHS parabolae have
vertices centered on the opposite side.

If $D_{xy} > 0$, the unregularized vertex is on the RHS.  Ramping $\lambda$
such that the RHS vertex $\left(2D_{xy} - \lambda\right)/(2S_x)$ is
pushed to the LHS means:

$$2D_{xy} - \lambda < 0 \Rightarrow \lambda > 2D_{xy}$$

If $D_{xy} < 0$, the unregularized vertex is on the LHS.  Ramping $\lambda$
such that $\left(2D_{xy} + \lambda\right)/(2S_x)$ falls on the RHS means:

$$2D_{xy} + \lambda > 0 \Rightarrow \lambda > -2D_{xy}$$

We can cover both these bases by saying, "compromise is ruled out and the
maximally-simple model dominates when:"

$$\lambda > 2|D_{xy}|$$

**This is the magic threshold for Lasso.**  Past this level of regularization,
we can only get the empty model out of the optimization.

Checking my work: in the blue-dot case, $D_{xy} = 314.7$, and the
"Regularization Path" figure shows that $w^*(\lambda)$ hits zero right where we
expect, around $2 \times 314.7 \approx 630$.

(It's good to know Lasso has this magic threshold: it means when tuning the
regularization parameter so that the model that drops out works on held-out
test data, we can put an upper bound on our search space.  That's useful!)

## Why's Lasso do that? (Discontinuous slope.)

The slope of our Lasso objective looks like:

$$\begin{align}\frac{d}{dw}\left\{f(w) + \lambda r(w)\right\} &= \frac{d}{dw}f(w) + \lambda\frac{d}{dw}\left|w\right| \\
&= \frac{d}{dw}\left\{S_xw^2 - 2D_{xy}w + S_y\right\} + + \lambda\frac{d}{dw}\left|w\right| \\
&= 2S_xw - 2D_{xy} + \lambda\frac{d}{dw}\left|w\right| \end{align}$$

The derivative for our regularizer hits a snag at $w = 0$; the slope tangent
to $|w|$ depends on whether you mean "sloping in from the LHS" or "sloping in
from the RHS":

$$\lambda\frac{d}{dw}|w| = \begin{cases}
          -\lambda&w < 0 \\
          \mbox{undefined}&w = 0 \\
          \lambda&w > 0
    \end{cases}$$

$$\frac{d}{dw}\left\{f(w) + \lambda r(w)\right\} = \begin{cases}
          2S_xw - 2D_{xy} - \lambda&w < 0 \\
          \mbox{undefined}&w = 0 \\
          2S_xw - 2D_{xy} + \lambda&w > 0
    \end{cases}$$

We can plot this for the zero-, some-, and lots-of-regularization models fit to
the blue-dot data, where we see a steadily larger discontinutiy at $w = 0$:

![A line plot titled "Slopes of lasso objectives" with an x-axis labelled "w"
and three series:
(1) in blue, a straight line with slope of 650 and intercept of -650 or so, with
a big blue dot marking its x intercept around 1.0;
(2) in purple, a piecewise linear function where both pieces have slope 650, but
a breakage at w=0, with intercepts of -1000 and 250 on either side, with a big
purple dot marking its x intercept around 0.49;
(3) in red, a piecewise linear function where both pieces have slope 650, but
a breakage at w=0, with intercepts of -1450 and 150](/images/0004_g_slopes.png)

For zero- and some-regularization, the blue and purple lines, we see
intersection with the horizontal axis at the same points we see the vertices of
their respective parabolae above: $w$ values where our Lasso objective has
zero slope.

But with lots-of-regularization, there's no similar intersection.  The
discontinuity introduced by $\lambda r(w)$ is large enough
that, at $w = 0$, the slope jumps straight from "very negative" to "slightly
positive." This pseudo-intersection with the horizontal axis determines where
our little-guy optimizer halts, just like the genuine points of zero slope in
the blue and purple models.

Looking at the slope discontinuity explains Lasso's magic threshold 
$\lambda > 2|D_{xy}|$ in a way that generalizes to other loss functions for $f$.
The regularizer is acting like a kind of crowbar, chiseled into the
unregularized objective at the point $(w, f'(w))$, and prying the LHS and RHS
apart from each other by an additive factor of $\lambda$.

Our blue slope line has a non-zero value at $w = 0$.  For $\lambda|w|$ to
introduce a sign change in the slope, it either needs to push a positive $f'(0)$
below the horizontal axis, or (as in the blue-dots example here) push a negative
$f'(0)$ above the horizontal axis.  As in, we get the maximally-simple model
whenever we set:

$$\lambda \geq \left|f'(0)\right|$$

For original-recipe Lasso , where our loss $f$ is the ordinary least squares
function, $f'(0) = 2D_{xy}$.  Plug that in above and recover the magic
threshold. Without any regularization (for $D_{xy} > 0$ here, but the logic
holds in the mirror case, too), $f(w)$ decreases as you move from $w = 0$ into
the RHS.  Even with *some* regularization, $f(w) + \lambda r(w)$ decreases as
you move into the RHS -- it has a negative slope at $w = 0^+$.

When $\lambda$ is big enough, the discontinuity of slope it introduces can
entirely swamp the initial slope $f(w)$ has, whether it points towards the LHS
or the RHS. That's why Lasso does that.


## Returning to multiple features

This same phenomenon applies even when we reinflate from $p = 1$ univariate
feature, back to the $p > 1$ multivariate case.  Lasso does this in high
dimensions, too.

Bringing back our original $N$ vector-scalar pairs,
$\left\{\vec{\mathbb{x}}_j, y_j\right\}_{j = 1}^N$, I'm going to define some
new helper aliases.  First, imagine stacking all the feature vectors,
transposed horizontally, into a matrix
$\mathbf{X} \in \mathbb{R}^{N \times p}$:

$$\mathbf{X} = \left[\begin{array}{c} ~~- \vec{\mathbb{x}}_1^T -~~ \\
\vdots \\
~~- \vec{\mathbb{x}}_j^T -~~ \\
\vdots \\
~~- \vec{\mathbb{x}}_N^T -~~ \\
\end{array}\right]$$

Now imagine pulling out the $i$th column from that matrix: it encodes the
$i$th predictive feature for each datum $j = 1, \ldots, N$.  Call each of these
column vectors $\vec{\mathbb{x}}^{(i)} \in \mathbb{R}^N$, $i = 1, \ldots, p$:

$$\mathbf{X} = \left[\begin{array}{c} ~~- \vec{\mathbb{x}}_1^T -~~ \\
\vdots \\
~~- \vec{\mathbb{x}}_j^T -~~ \\
\vdots \\
~~- \vec{\mathbb{x}}_N^T -~~ \\
\end{array}\right] = \left[\begin{array}{ccccc}
\vert & & \vert & & \vert \\
\vec{\mathbb{x}}^{(1)} & \cdots & \vec{\mathbb{x}}^{(i)} & \cdots & \vec{\mathbb{x}}^{(p)} \\
\vert & & \vert & & \vert \\
\end{array}\right]$$

And let's also pull the labels $y_j$ into their own column vector:

$$\vec{\mathbb{y}} = \left[\begin{array}{c} y_1 \\
\vdots \\
y_N
\end{array}\right] \in \mathbb{R}^N$$

### Full sparsity

Imagine, for our multivariate dataset, we have currently set our little-guy
optimizer at the fully sparse, all-zeros weight vector,
$\vec{\mathbb{w}} = \vec{\mathbb{0}}$.  What would it take to convince the
little guy to move any single model weight off of zero?

The little guy can consider each feature, $\vec{\mathbb{x}}^{(i)}$, one at a
time, as if each were its own individual univariate Lasso case.  The current
weight vector ignores *every* feature, so the question of "should we put any
weight on feature $i$?" depends only on the univariate dataset
$\left\{\vec{\mathbb{x}}^{(i)}_j, y_j\right\}_{j=1}^N$; the other features are
currently zeroed out and of no concern.

For feature $i$, the magic threshold is the analogue of $2|D_{xy}|$, taking a
dot product between the $i$th feature column vector and the label column vector:

$$\lambda > 2\left|\vec{\mathbb{y}}^T\vec{\mathbb{x}}^{(i)}\right|$$

The little guy will not move off $\vec{\mathbb{w}} = \vec{\mathbb{0}}$ if
$\lambda$ is big enough to cross the magic threshold for all $p$ features.
Which means have a *meta*-magic threshold in the multivariate case that zeros
out all $p$ model weights, just by ramping $\lambda$ to a big enough value:

$$\lambda > \max_{i = 1, \ldots, p}\left|\vec{\mathbb{y}}^T\vec{\mathbb{x}}^{(i)}\right|
\Rightarrow \vec{\mathbb{w}}^*(\lambda) = \vec{\mathbb{0}}$$

(Note that the $\max_i$ operation is the same as, for our definition of matrix
$\mathbf{X}$ above: "calculate the vector
$\mathbf{X}^T\vec{\mathbb{y}}$, then find the largest-magnitude element.")

This is also good to know!  If we're using crossvalidation to hunt for the
perfect $\lambda$ parameter, it continues to be nice to have an upper bound on
the search space.

### Partial sparsity

Unregularized multivariate ordinary least squares will, in almost all cases,
give you a $w^*(0)$ with weight on every feature.  And if we ramp $\lambda$
past the meta-magic threshold, we can get a fully sparse, all-zeros set of
model weights, $\vec{\mathbb{w}}^*(\lambda_{\text{lots}}) = \vec{\mathbb{0}}$.
So it stands to reason that somewhere in along the ramp between "no sparsity"
and "full sparisty" you get values of $\lambda$ that mean "some sparsity."

For example, if you set $\lambda$ *juuuust* below the meta-magic threshold,
then whatever feature triggered the max,
$i = \arg \max_{i' = 1, \ldots, p}\left|\vec{\mathbb{y}}^T\vec{\mathbb{x}}^{(i')}\right|$,
will be non-zero, but none of the other weights will have turned on yet.

It's hard to say what happens after that.  The dynamics depend on the covariance
statistics you see between your features across the $N$ data points.  But we
can at least talk about what it means for our little guy to settle in on a
partially sparse solution.