"""experiments.yaml 읽어서 현재 상황 출력."""
import yaml

def main():
    with open('experiments.yaml', 'r') as f:
        experiments = yaml.safe_load(f)

    print(f"{'Phase':<15} {'CV':>8} {'Public':>10} {'Rank':>5} {'Notes'}")
    print("-" * 70)

    for exp in experiments:
        cv = exp.get('cv', '-')
        public = exp.get('public', '-')
        rank = exp.get('rank', '-')
        notes = exp.get('notes', '')[:40]

        if cv and cv != '[unknown]':
            cv_str = f"{cv:.4f}" if isinstance(cv, (int, float)) else str(cv)
        else:
            cv_str = '-'

        if public and public != '[unknown]':
            pub_str = f"{public:.5f}" if isinstance(public, (int, float)) else str(public)
        else:
            pub_str = '-'

        rank_str = str(rank) if rank else '-'

        print(f"{exp['phase']:<15} {cv_str:>8} {pub_str:>10} {rank_str:>5} {notes}")

    # Best
    best = max(
        [e for e in experiments if isinstance(e.get('public'), (int, float))],
        key=lambda x: -x['public'],
        default=None
    )
    if best:
        print(f"\nCurrent best: {best['phase']} (Public {best['public']}, Rank {best.get('rank', '?')})")

if __name__ == '__main__':
    main()
