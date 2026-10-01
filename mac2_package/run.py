"""
Entry point. Usage:
    python run.py                          # runs the example below
    python -c "from mac2 import run_experiment; run_experiment('Cora')"
"""
from mac2 import run_experiment

if __name__ == '__main__':
    # Single run
    res = run_experiment('Cora', epochs=100, gl_operator='weighted', lambda_gl=0.1,
                     output_dir='./outputs')
    # print(res)

    # ── Ablation example (uncomment to run all three) ──
    # for cfg in [
    #     dict(gl_operator='structural', lambda_gl=0.0),   # baseline (GL off)
    #     dict(gl_operator='structural', lambda_gl=0.1),   # GL, homophily operator
    #     dict(gl_operator='weighted',   lambda_gl=0.1),   # GL, heterophily-aware
    # ]:
    #     run_experiment('Roman-empire', epochs=100, **cfg)
