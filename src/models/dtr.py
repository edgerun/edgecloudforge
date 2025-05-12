from src.motivationalhetero.encoders import get_platform_type_encoder
from src.train import  \
    train_dtr_model_reactive_post_experiment_then_proactive_per_device_type


def main():
    results = [
        'globus-endpoint-02-output-40/infra-1/results-reactive/20250505-201512-773281/0/0',
        'globus-endpoint-02-output-40/infra-1/results-reactive/20250505-201512-773281/0/1',
        'globus-endpoint-02-output-40/infra-1/results-reactive/20250505-201512-773281/0/2',
        'globus-endpoint-02-output-40/infra-1/results-reactive/20250505-201512-773281/0/3',
    ]
    # with open(results_file, 'r') as fd:
    #     results = json.load(fd)
    # print(len(results))
    encoder = get_platform_type_encoder()

    train_dtr_model_reactive_post_experiment_then_proactive_per_device_type(results, False, encoder=encoder)


if __name__ == '__main__':
    main()
