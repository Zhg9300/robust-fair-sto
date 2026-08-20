import components as cn
import os

if __name__ == '__main__':

    params = cn.read_params()
    data_loader, algorithm = cn.initialize(params)
    algorithm.save_folder = 'results/' + data_loader.nickname + '/' + params['module'] + '/' + params['algorithm'] + '/'
    if not os.path.exists(algorithm.save_folder):
        os.makedirs(algorithm.save_folder)
    attack_suffix = algorithm.attack_controller.experiment_suffix()
    run_prefix = 'seed' + str(params['seed']) + ' N' + str(data_loader.pool_size) + ' R' + str(params['R'])
    if attack_suffix:
        run_prefix += ' ' + attack_suffix
    algorithm.save_name = run_prefix + ' ' + algorithm.save_name
    algorithm.run()
