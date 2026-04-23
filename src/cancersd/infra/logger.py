import sys
import time


def format_model_info(model_dict, hyper_parameters=None, performance=None, log_file='screenshot.log'):
    temp = sys.stdout
    log = open(log_file, 'a')
    sys.stdout = log

    headers = ['name', 'layer', 'parameters']
    current = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f'model structure, start time->{current}')
    print(f'{headers[0]:<32}{headers[1]:<12}{headers[2]:<20}')

    for k, config in model_dict.items():
        for index, params in enumerate(config):
            layer, param = params[0], str(params[1])
            if index == 0:
                print(f'{k:<32}{layer:<12}{param:<20}')
            else:
                print(f'{" ":<32}{layer:<12}{param:<20}')

    if hyper_parameters is not None:
        for k, v in hyper_parameters.items():
            print(f'{k} -> {v}')

    if performance is not None:
        for k, v in performance.items():
            print(f'{k} -> {v}')

    print()

    sys.stdout = temp
    log.close()