import json

import pandas as pd


def main():
    path = 'globus-endpoint-02-output-20/first_second_third/infra-1/results-proactive/R1-49-origin-1-target-1/20250506-184126-302465/0/0'
    print('123pro')
    count_stats(path)
    path = 'globus-endpoint-02-output-20/first_second/infra-1/results-proactive/R1-49-origin-1-target-1/20250506-184126-422930/0/0'
    print('12pro')
    count_stats(path)
    path = 'baas/infra-1/results-proactive/abdullasynth/resource_model.joblib-origin-1-target-1/20250506-165219-642609/0/3'
    print('baas')
    count_stats(path)


def count_stats(folder):
    system_event_path = f'{folder}/system_events.csv'
    system_event_df = pd.read_csv(system_event_path)
    print(system_event_df['count'].mean())
    print(system_event_df['count'].max())
    print(system_event_df['count'].median())
    print(system_event_df['count'].std())

    results_path = f'{folder}/results.json'
    with open(results_path, 'r') as fd:
        results = json.load(fd)
    energy = results['energy']
    print(f'Energy {energy}')

if __name__ == '__main__':
    main()
