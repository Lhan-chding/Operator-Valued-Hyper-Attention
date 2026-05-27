# OVHA Context Identifiability

## Metadata-Free Episode Definition

Phase 1.5 uses metadata-free episodic meta-operator learning. Each episode samples a latent task `tau` and an operator:

`G_tau: U -> Y^Q`.

The model observes only:

- `context_u`: context input functions sampled on support coordinates.
- `context_q`: context query coordinates.
- `context_y`: observed outputs `G_tau(u_m)(q_mj)`.
- `target_u`: target input function.
- `target_q`: target query coordinates.
- `support_grid`: support coordinates.
- optional masks.

The model must not receive `family`, `operator_id`, `gain`, `latent_params`, `mixture_weights`, `oracle_hints`, or primitive weights.

## Episode Modes

`same_function_field` is the weaker field-completion setting: context and target come from the same input function field. It is useful for debugging query interpolation and routing.

`operator_transfer` is the primary Phase 1.5 setting: context contains several input-output demonstrations, and the target uses a new input function `u*`. The model must infer the latent operator from context observations before applying it to `u*`.

## Identifiability Condition

A context design `D_n` is `gamma`-identifying for task family `T` if, for any `tau_1 != tau_2`, at least one observed context pair separates the operators:

`||G_tau1(u_m)(q_mj) - G_tau2(u_m)(q_mj)|| >= gamma d_T(tau_1, tau_2)`.

A weaker useful condition is injectivity up to `epsilon`:

`tau -> observations(D_n; tau)` is injective up to `epsilon`.

This condition is necessary. If two operators are identical on the observed context but differ at the target query, no metadata-free learner can reliably solve the episode.

## Confusable Context Impossibility

Phase 1.5 should include `confusable_context` splits where the context covers too few or too narrow query regions. Failure on this split is not a method failure by itself; it is evidence that the evaluation respects the information-theoretic boundary.

The report must not claim success on confusable episodes unless the observations actually distinguish the latent operator.

## Universal Approximation Sketch

Assume:

1. Task space `T` is compact.
2. Input function set `U` is compact under the chosen norm.
3. Query domain `Q` is compact.
4. The metadata-free map `(D_n, u*, q) -> G_tau(u*)(q)` is continuous on the identifiable subset.
5. Memory encoder, router and hyper-adapter are universal approximators on compact sets.
6. The primitive registry contains a dense primitive closure for the target operator family, or includes a universal fallback primitive.

Then, for every `epsilon > 0`, there exists an OVHA model such that:

`sup_{tau,u*,q} ||F_theta(D_n,u*,q) - G_tau(u*)(q)|| < epsilon`

on the identifiable subset.

The statement must not be read as unconditional. It depends on context identifiability and primitive closure.

## Discretization Consistency

If each primitive kernel `K_r(q,s;eta_r)` is continuous or integrable and support samples approximate the domain by quadrature, then:

`sum_j w_j K(q,s_j)u(s_j) -> integral K(q,s)u(s)ds`.

Finite primitive mixtures preserve this convergence under bounded router weights. Phase 1.5 therefore includes a `resolution_transfer` split: train on lower support resolution and evaluate on higher resolution.
