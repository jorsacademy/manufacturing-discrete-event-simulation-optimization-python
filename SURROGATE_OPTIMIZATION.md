# Neural Surrogate-Assisted Simulation Optimization

The base repository deliberately enumerates a small declared 50-design grid. That is the correct method for that small search space. This extension asks a different question: **what happens when the design space is enlarged enough that exhaustive replicated simulation is expensive?**

The extended search space can contain thousands of combinations of two buffer capacities and repair-technician staffing. A neural network learns

[
(buffer_1, buffer_2, technicians) mapsto E[profit/hour]
]

from a limited set of discrete-event simulation evaluations.

The workflow is:

```text
initial space-filling designs
        ↓
common-random-number DES evaluations
        ↓
neural profit surrogate
        ↓
predict all unevaluated designs
        ↓
select promising batch
        ↓
true DES evaluation
        ↓
refit and repeat
```

The final candidate is validated with independent seeds. A random-search baseline receives the **same number of evaluated designs and the same replication budget**.

This is intentionally not presented as a replacement for exhaustive enumeration when the original 50-point grid is affordable. The surrogate becomes relevant only when the design space, simulator runtime, replication requirement, or number of decision variables grows.

Install the neural extension with:

```bash
pip install -r requirements-surrogate.txt
python neural_surrogate_optimization.py
```
