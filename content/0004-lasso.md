Title: Why's Lasso Do That?
Date: 2025-08-14 11:30
Slug: 0004-lasso
Category: stats
Tags: stats, ml, lasso
Author: Brian Gawalt
Summary: How the Lasso forces a zero-weight model: an end-to-end rundown.
opengraph_image: 0004_lasso_twirl.png

(**Note:** this post makes heavy use of MathJax. If you're reading via RSS,
you'll want to click through to the web version.)

TODO:

* Ridge matplotlib
* Lasso matplotlib
* Interpret lasso results
* Multivariateness

![MS Paint doodle of a purple lambda (the Greek alphabet character) twirling a lasso in the desert](/images/0004_lasso_twirl.png){: style="width:80%; max-width:500px;"}

My intuition of regularization it's a compromise.  You want a model that fits
your historical examples, but you also want a model that is "simple."
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
by regularizing a model fit.  It's not a compromise anymore, and I'd like to
unspool what exactly's driving this blunt outcome. Why's Lasso do that?

## What's Lasso do? (Parabolas.)

### Zero regularization

### Some regularization

### Lots of regularization

## Why's Lasso do that? (Discontinuous slope.)

## Unregularized linear regression

In the univariate regression case, we have one scalar input that describes an
example (like a house's lot size), which we're using to predict that
example's label (like the house's sale price).  We have some collection of
labeled examples, where the $i$th example is described by the pair of scalars
$(x_i, y_i)$, that we'll use to learn our linear model.

Unregularized linear regression defines its cost function as:

$$\mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) = \sum_{i=1}^N \left(y_i - w x_i\right)^2$$

(The $(U)$ superscript is just me denoting "**u**nregularized.")

That cost changes as a function of the model weight $w$.  Picking a particular
model weight value implies an interpretation, "when a house's lot size is one
$x$-unit larger (i.e., one additional square foot), its sale price is probably
$w$ $y$-units larger (i.e., $w$ additional US dollars)."

Better-fit models are the ones where $w$ pushes the product $w x_i$
closer to $y_i$, on average.  And so the traditional way to pick a good value
for $w$ is to minimize a cost function that captures that criteria:

$$\begin{align}w^{(U)} &= \arg \min_w \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right)\\
 &= \arg \min_w \sum_{i=1}^N \left(y_i - w x_i\right)^2\end{align}$$

This minimizing weight value is the one for which the slope
of $\mathcal{L}^{(U)}$ is zero, familiar from regular old high school calculus:

* $\mathcal{L}^{(U)}$ is *convex*,
* meaning its slope is never decreasing in $w$,
* so the point of zero slope is the exact value of $w$ where $\mathcal{L}^{(U)}$
  changes from "the function is decreasing" to "the function is increasing,"
* which means we've found the lowest point on $\mathcal{L}^{(U)}$.

Calculating $\mathcal{L}^{(U)}$'s slope as a function of $w$ is easier with some
terms rearranged:

$$\begin{align}\mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) &= \sum_{i=1}^N \left(y_i - w x_i\right)^2 \\
&= \sum_{i=1}^N \left[y_i^2 - 2 x_i y_i w + x_i^2 w^2\right] \\
&= \left[\sum_{i=1}^N y_i^2\right] - 2\left[\sum_{i=1}^N (x_i y_i)\right] w + \left[\sum_{i=1}^N x_i^2\right] w^2\end{align}$$

Taking the derivative of $\mathcal{L}^{(U)}$ with respect to $w$ gives:

$$\begin{align}\left.\mathcal{L}^{(U)}\right.'(w; \{y_i, x_i\}) &= \frac{d}{dw}\left\{\sum_{i=1}^N y_i^2\right\} -
               \frac{d}{dw}\left\{2\left[\sum_{i=1}^N (x_i y_i)\right] w\right\} + 
               \frac{d}{dw}\left\{\left[\sum_{i=1}^N x_i^2\right] w^2\right\} \\
         &= 0
               - 2\left[\sum_{i=1}^N (x_i y_i)\right]
               + 2\left[\sum_{i=1}^N x_i^2\right] w \\
         &= 2\left[\sum_{i=1}^N x_i^2\right] w 
               - 2\left[\sum_{i=1}^N (x_i y_i)\right] \end{align}$$

Per above: at our minimizing weight value, this derivative is zero.  Call the 
weight value associated with zero-slope to be $w^{(U)}$, and find it:

$$\left.\mathcal{L}^{(U)}\right.'(w^{(U)}; \{y_i, x_i\}) = 0 \Rightarrow
2\left[\sum_{i=1}^N x_i^2\right] w  - 2\left[\sum_{i=1}^N (x_i y_i)\right] = 0$$

$$w^{(U)} = \frac{\sum_{i=1}^N (x_i y_i)}{\sum_{i=1}^N x_i^2}$$

For unregularized univariate regression, the model weight we select is the
ratio of "sum of each feature times its associated label" over "sum of each
feature times itself." Because we'll use them a lot later, let's give these
two terms their own alias definitions:

$$D := \sum_{i=1}^N (x_i y_i)$$

where $D$ is for "dot," cuz it's the dot product between two $N$-dimentional 
vectors, one vector holding each example's input/feature scalar and one holding
each example's label scalar. And:

$$S := \sum_{i=1}^N x_i^2$$

where $S$ is for "**s**um of **s**quares."

Plugging the aliases in to the two equations above, our unregularized cost
function and its derivative are:

$$\mathcal{L}^{(U)} = Sw^2 - 2Dw + \left[\sum_{i=1}^N y_i^2\right]$$

$$\left.\mathcal{L}^{(U)}\right.'(w; \{y_i, x_i\}) = 2Sw - 2D$$

and that derivative crosses zero at:

$$w^{(U)} = \frac{D}{S}$$


## Ridge regression

When we minimized $\mathcal{L}^{(U)}$, we were picking a $w$ that
matched the predictions to the actual labels across the historical training
data.  Regularization adds a new term to that cost function that says, "a good
weight parameter will fit the data, but will also have
\[other desired property of a model weight, orthogonal to its fit to the
data\]."  And in my intuition, it should be a graceful compromise: find a way to
meet in the middle between data-fitting vs. expressing the newly-introduced
orthogonal property.

Ridge regression is a type of regularization that matches that intuition.
We can start with $\mathcal{L}^{(U)}$ as the data-fit half of the mixed
objective, and then add a term to the cost function that asks that the weight we
find be close to zero. (See a postscript below, "Why hedge towards zero?")

Ridge regression implements "hedge towards zero" regularization by adding
a quadratic penalty term.  Call the ridge regression cost
function $\mathcal{L}^{(R)}$, where the superscript $(R)$ means **r**idge:

$$\mathcal{L}^{(R)}\left(w; \{y_i, x_i\}\right) = \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) + \mu w^2$$

Two components, mixed into one objective: our original data-fit objective, plus
the regularization term $w^2$, as adjusted by a scale factor $\mu$. The data-fit
term wants $w$ to be $\left(D/S\right)$; the regularizer term wants it to be
zero.  Except in the rare case where $D$ is itself zero, these conflict.  The
scalar hyperparameter $\mu$, which we always set to a non-negative value, acts
as the "exchange rate" between these two desiderata: the bigger $\mu$, the more
we pull our regularized model weight away from $w^{(U)}$ and towards zero.

As before, we'll fit the model by finding the weight value that minimizes
$\mathcal{L}^{(R)}$, which is the same as the one for which
$\left.\mathcal{L}^{(R)}\right.'$ is zero:

$$\begin{align}\mathcal{L}^{(R)}\left(w; \{y_i, x_i\}\right) &= \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) + \mu w^2\\
\left.\mathcal{L}^{(R)}\right.'\left(w; \{y_i, x_i\}\right) &= \left.\mathcal{L}^{(U)}\right.'\left(w; \{y_i, x_i\}\right) + \frac{d}{dw}\left\{\mu w^2\right\} \\
&= 2Sw - 2D + 2\mu w\\
&= 2\left(\mu + S\right)w - 2D\end{align}$$

At the minimum of the ridge objective, we have:

$$\left.\mathcal{L}^{(R)}\right.'\left(w^{(R)}; y, x\right) = 0 \Rightarrow 2\left(\mu + S\right)w^{(R)} - 2D = 0$$

$$w^{(R)} = \frac{D}{\mu + S}$$

When $\mu$ is zero, we recover $w^{(R)} = D/S$, same as $w^{(U)}$.  When $\mu$
is very large, $w^{(R)}$ gets very close to zero.  *Close!*  Never *exactly*
zero.  It's always tugged, at least a little bit, in the direction of $w^{(U)}$.

This is the pleasant, intuitive, compromise dynamic.  When $\mu$ is positive,
we always land somewhere *between* "optimally fit the data" and "be zero."
Both terms in the mixed objective get their say.


## The Lasso

There's another way to encode "be close to zero" as a regularizer.  Instead of
the quadratic $w^2$ penalty of ridge, we could instead use $w$'s absolute value.
This is called the Lasso, with cost function $\mathcal{L}^{(L)}$:

$$\mathcal{L}^{(L)}\left(w; y, x\right) = \mathcal{L}^{(U)}\left(w; y, x\right) + \lambda|w|$$

where $\lambda$ is now our non-negative penalty-rate hyperparameter.

To find the minimum of this cost function, we can try repeat the strategy that
worked for $\mathcal{L}^{(U)}$ and $\mathcal{L}^{(R))}$, "take the
derivative and find a value $w^{(L)}$ that makes it zero." But watch what
happens when we do:

$$\begin{align}\left.\mathcal{L}^{(L)}\right.'\left(w; \{y_i, x_i\}\right) &= \left.\mathcal{L}^{(U)}\right.'\left(w; \{y_i, x_i\}\right) + \frac{d}{dw}\left\{\lambda |w|\right\} \\
&= 2Sw - 2D + \lambda \frac{d}{dw}\left\{|w|\right\}\end{align}$$

We've got a function, $\left.\mathcal{L}^{(L)}\right.'$, that's taking the
derivative of the absolute value function.  That derivative yields one of three
conditions:

$$\frac{d}{dw}\left\{|w|\right\} = \left\{\begin{array}{ll} +1&;~w > 0 \\ -1&;~w < 0 \\ \mbox{undefined}&;~w = 0 \end{array} \right.$$

So the derivative of $\mathcal{L}^{(L)}$ is defined for positive and negative
weight values, but not when $w$ is exactly zero. 

### Lasso's minimum is a positive weight

Assume that $\lambda$, $S$, and $D$ are setting us up to find that
$\left.\mathcal{L}^{(L)}\right.'\left(w^{(L)}; \{y_i, x_i\}\right)$ is
well-defined to have a value of zero for a strictly positive $w^{(L)}$.

This means that the derivative of $|w|$ must be well-defined to be 1, so:

$$\begin{align}\left.\mathcal{L}^{(L)}\right.'\left(w^{(L)}; \{y_i, x_i\}\right) = 0 &\Rightarrow 2Sw^{(L)} - 2D + \lambda \cdot 1 = 0 \\
&\Rightarrow w^{(L)} = \frac{2D - \lambda}{2S} = \frac{D - \lambda/2}{S}\end{align}$$

This value is less than $w^{(U)} = D/S$.  Lasso has shrunk our model parameter
towards zero.  Compromise!

Since the denominator $S$ is always positive (it's a sum of squares), for this to
hang together with our $w^{(L)} > 0$ assumption (and our $\lambda \geq 0$
definition), we need a positive numerator:

$$2D - \lambda > 0 \Rightarrow 0 \leq \lambda < 2D$$

So when $D$ is positive, and larger than $\lambda/2$ (i.e., we don't regularize
the estimation too hard), we'll get a nice compromise between $D/S$ and zero.

If we do regularize too hard -- i.e., if $\lambda > 2D$ -- we lose coherence.
A founding assumption was that $w^{(L)}$ is positive, but our formula of
$w^{(L)} = \frac{D - \lambda/2}{S}$ yields a negative value. A contradiction.
If $\lambda$ is too big, this formula for the minimal value is incorrect.

### Lasso's minimum is a negative weight

Quite similar to the got-a-positive-weight scenario, the derivative of $|w|$
is also assumed to be well-defined, this time at a value of -1:

$$\begin{align}\left.\mathcal{L}^{(L)}\right.'\left(w^{(L)}; \{y_i, x_i\}\right) = 0 &\Rightarrow 2Sw^{(L)} - 2D + \lambda \cdot (-1) = 0 \\
&\Rightarrow w^{(L)} = \frac{2D + \lambda}{2S} = \frac{D + \lambda/2}{S}\end{align}$$

The value $w^{(L)}$ is a shift away from $D/S$ and towards zero, just like the
previous scenario.  To make this coherent, though, we require that numerator
term to be negative:

$$2D + \lambda < 0 \Rightarrow 0 \leq \lambda < -2D$$

So again: if $D$ is negative, and we don't regularize too hard, we'll land in
this cool scenario, with a weight value somewhere between "fit the data" and
"be zero." Regularize too much, and this scenario is incoherent.

### Lasso's minimum is exactly zero

Finally, the uncool outcome.  What must be true about $\lambda$, relative to 
$S$ and $D$, to make $w^{(L)}$ exactly zero?  The cost function's derivative is
undefined for $w = 0$, and so our approach in every previous scenario can't
apply here:

$$\begin{align}\left.\mathcal{L}^{(L)}\right.'\left(0; \{y_i, x_i\}\right) &= \left.\mathcal{L}^{(U)}\right.'\left(0; \{y_i, x_i\}\right) + \left.\frac{d}{dw}\left\{\lambda |w|\right\}\right|_{w = 0} \\
&= \left(2S \cdot 0 - 2D\right) + \lambda  \left.\frac{d}{dw}\left\{|w|\right\}\right|_{w = 0} \\
&= -2D + \mbox{undefined} \\
&= \mbox{undefined}\end{align}$$

With the cost function's derivative undefined, we can't work backwards to decide
what $\lambda$ would need to be relative to $D$ and $S$ to make that defined value zero.

#### The subderivative

To address this kind of convex function, where the derivative is undefined, the
we have instead the concept of a
[subderivative](https://en.wikipedia.org/wiki/Subderivative).  It's a
generalization of the derivative: "if a line intersects our convex function
$f$ at some point, without any point on that line exceeding the value of $f$,
is the range of slopes that line could take?" The subderivative returns a *set*
of slope values at any point on your function that satisfy this "don't cross
over the function" criteria.

For any point where $f$ is differentiable, $f$'s subderivative is pleasantly
boring. The range of allowable slopes is a singleton: it's just the derivative
of $f$. That's what it means for a line to be tangent to a convex function.
That special and unique tangent line intersects the convex function, but never
crosses it.

For points where $f$ is not differentiable, the subderivative is a range of
values. In the absolute value case, $f(x) = |x|$, that range is $[-1, 1]$.
Here's $f(x) = |x|$ and a buncha lines whose slopes are in the subderivative
of $f$ at the point $x = 0$:

![Matplotlib line chart of f(x) = |x| in a solid blue line plus smaller blue dashed lines that pass through (0, 0) but never exceed |x|](/images/0004_abs_subdiff.png)

The absolute value function is convex, and has a unique global minimum. And we
can see that the unique global minimum is at $x = 0$. That's where the function
flips from decreasing to increasing. The subderivative at $x = 0$ includes
positive slopes, negative slopes, and -- hey! -- the zero slope. (At every other
value of $x$, the subderivative is either the singleton $\{+1\}$ or the
singleton $\{-1\}$.)

We can now give ourselves an upgrade from the derivative-based minimization
technique "from regular old high school calculus," to one backed by
subderivatives. Before, we wanted the point at which the derivative was zero.
Now, we want a point where the subderivative *contains* zero.

#### Lasso's subderivative

Let's evaluate the subderivative of $\mathcal{L}^{(L)}$ at $w = 0$. To do this,
we'll need to apply a linearity property of subderivatives.  The subderivative
of $f$ plus $g$, is
[the Minkowski sum](https://en.wikipedia.org/wiki/Minkowski_addition) of $f$'s
subderivative and $g$'s subderivative. The Minkowski sum is a way to combine two
sets of addable elements into a third set: take every pairwise combo, where the
first element is from the first set and the second from the second, and include
their sum as an element in that output set.

If we make $\partial$ our operator symbol for "take the subderivative," and
$\oplus_M$ to mean "take the Minkowski sum", we can start evaluating the
subderivative of $\mathcal{L}^{(L)}$ at $w = 0$:

$$\begin{align}\partial\left\{\mathcal{L}^{(L)}\left(w; \{y_i, x_i\}\right)\right\} &= \partial\left\{ \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) + \lambda |w| \right\} \\
&= \partial\left\{ \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) \right\} \oplus_M \partial\left\{ \lambda |w| \right\}\end{align}$$

The left term is taking the subderivative of $\mathcal{L}^{(U)}$. And that's
easy: $\mathcal{L}^{(U)}$ is everywhere-differentiable, so its subderivative is
everywhere just a singleton whose lone element is
$\left.\mathcal{L}^{(U)}\right.'$.  The right term is the subderivative of a
scaled-up absolute value function.

If we evaluate at $w = 0$:

$$\begin{align}\left.\partial\left\{\mathcal{L}^{(L)}\left(w; \{y_i, x_i\}\right)\right\}\right|_{w = 0} &= \left.\partial\left\{ \mathcal{L}^{(U)}\left(w; \{y_i, x_i\}\right) \right\}\right|_{w = 0} \oplus_M \left.\partial\left\{ \lambda |w| \right\}\right|_{w=0} \\
&= \left\{ \left.\left.\mathcal{L}^{(U)}\right.'\left(w; \{y_i, x_i\}\right)\right|_{w=0} \right\} \oplus_M \left\{ \omega~|~\omega \in [-\lambda, \lambda]\right\} \\
&= \left\{ \left.\left(2Sw - 2D\right)\right|_{w=0} \right\} \oplus_M \left\{ \omega~|~\omega \in [-\lambda, \lambda]\right\} \\
&= \left\{ -2D \right\} \oplus_M \left\{ \omega~|~\omega \in [-\lambda, \lambda]\right\} \\
&= \left\{ \omega - 2D~|~\omega \in [-\lambda, \lambda]\right\}\end{align}$$

Which is to say, the subderivative of $\mathcal{L}^{(L)}$ at $w=0$ is the
set of slopes in the range $[-\lambda - 2D, \lambda - 2D]$.

For $w^{(L)}$ to be zero, we need the zero-slope to be in that range:

$$\begin{align}0 \in~& [-\lambda - 2D, \lambda - 2D] \\
\Rightarrow~&-\lambda - 2D \leq 0 \leq \lambda - 2D \\
\Rightarrow~& -\lambda \leq 2D \leq \lambda \\
\Rightarrow~& \lambda \geq |2D| \end{align}$$

We now have our conditions under which the Lasso model's weight $w^{(L)}$ is
zero:

> **When $\lambda \geq 2\sum_{i=1}^N (x_i y_i)$, the univariate Lasso gives a
> pure zero model weight.**

This concurs with our earlier two constraints on when $w^{(L)}$ could be
positive or negative.  We only get a non-zero $w^{(L)}$ when $\lambda$ is
small relative to $|D|$.


## Why's Lasso do that?

To better understand why a sufficiently large $\lambda$ turns Lasso from a
shrinkage operation to a *vanishing* operation, let's draw some pictures.

First, we'll simplify things by just assuming that $D$ is positive. The
intuition will still hold for the $D$-is-negative case, too; we'll just need to
mentally flip stuff across the vertical axis.


## Lasso in multiple dimensions

It's a convex objective.


## Postscript: Why hedge towards zero?

Both ridge and the Lasso use "the weight is close to zero" as the "orthogonal
property" that the regularized model combines with the basic fit-the-data
objective.

Using that as the competing goal of the regularized regression is a way of
baking humility into our model fit.  Imagine $x$ and $y$ are of a similar scale.
Maybe because you just picked units that are a natural fit for this (in the Bay
Area, house floor area in units of "square feet," and price in units of
"thousands of USD"). Or maybe, you've
[z-score-standardized](https://en.wikipedia.org/wiki/Standard_score)
your raw data to be zero-mean and unit-variance.  When $x$ and $y$ are on
the same  basic scale, there are two ways for $w^{(U)}$ to be far from zero:

1.  You've stumbled upon a relationship between $x$ and $y$ where even the
    tiniest increase in $x$ means a giant swing in $y$.  That happens sometimes!
    Usually you don't need statistical analysis to notice those kinds of
    effects, though.
2.  You've just got some noise in your data that's caused you to overestimate
    the effect of $x$ on $y$.  This also happens sometimes.

So "be close to zero" is a way to say "if you're seeing a large effect size
in terms of $D$-over-$S$, maybe hedge that towards zero; maybe that hedge is
washing out a genuine law of nature, but you're probably washing out noise
instead."  Humility.