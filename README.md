## High-Level Concept

This is a package for efficiently implementing household stages of the following kind.

Primitive household stages are fundamentally one of the following: 
- softmax with temperature between 0 and 1 (exogenous-markov and hard-argmax are the endpoint cases)
- affine shifts backward or forward
- pointwise multiplication backward or forward
which operate on one axis of household heterogeneity at a time.

Household stages in general are the finite span of primitive stages under the composition and product operators.

## Modular Implementation

In order to implement these, we require some general utilities for data representation, "stratification" (operating on one axis at a time, but possibly in ways that depend on the coordinates along other axes), and "generalized broadcasting" (applying stratified operations on an entire array at once). Then we can use those utilities to build and define kernels and primitive stages. We also need to implement compositions and products of stages. Finally, we can use those primitive stages to define derived stages.

Maintaining modularity and separation of these concerns is crucial for maintainability, transparency, and performance.

Therefore, the contents of src/ are as follows:

1. Layout utilities which instantiate multi-axis household state spaces, with Symbol-labeled axes, in a specified order, which can be a mixture of discrete and gridded-continuous.
2. Stratification utilities. In a sense, these can be thought of as generalizing Julia's "broadcasting". These consist of:
    a. Utilities for defining scalar-valued fields that possibly vary by axis.
    b. Utilities for defining matrix-valued fields which specify their "operative axis", that possibly vary by the **other** axes (axes other than the operative axis).
    c. Utilities for allocating, filling, caching, and refreshing the above given an `env`.
    d. Utilities for broadcasting a "(matrix, vector) -> vector)" closure, and its in-place version, over a matrix-valued field. That is, applying each matrix-valued slice of the matrix-valued field over each vector-valued slice of an array in a gridded layout to be operated on.
    e. Utilities for broadcasting a "(scalar, vector) -> vector)" closure, and its in-place version, over a scalar-valued field. That is, applying each scalar-valued slice of the scalar-value field over each vector-valued slice of an array in a gridded layout to be operated on. (See mean-preserving spread example.)
3. Kernel-application utilities. These include:
    a. Utilities for efficiently applying a matrix-valued field over its operative axis as a stratified tensor contraction, in either covariant/original or contravariant/transpose mode.
    b. Utilities for applying integer-valued scalar-valued fields as scattering along a discrete axis, also in original or transpose mode.
    c. Utilities for applying float-valued scalar-valued fields as scattering along a continuous axis.
4. Utilities for defining "moments" (integrals) as closures, and computing them given `env` and \Lambda.
5. Primitive stages:
    a. The pointwise primitive stages, hard argmax, softmax, and Markov.
    b. A few other optimization problems over slightly more exotic subspaces of big Hilbert spaces, such as continuous argmax, optimal choice of mean-preserving spread with costly effort, and bias-variance trade-off. The reason these cases are handled by separate primitive stages is for speed and auto-differentiability: the solutions to these optimization problems can be stored as per-cell floats, which are differentiable, rather than via non-differentiable references to specific destination-indices.
6. The composition operator.
7. The product operator.
8. Derived stages, consisting of composed/producted or wrapped primitive stages.

## Examples

In `examples/`, many models from the literature are constructed by building the household blocks using the HouseholdStages package. A catalogue of them, and more detailed explanation, is available in `EXAMPLES.md`

**Note: Much of `EXAMPLES.md` and the model implementations in `examples/` were constructed by language models.**

