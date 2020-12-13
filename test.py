import environment
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data params

    parser.add_argument('-group_num', default=0,type=int)
    parser.add_argument('-scene_num', default=0,type=int)
    args = parser.parse_args()


    scene_name = 'group-'+ '0'+str(args.group_num) + '-scene-'+'0'+ str(args.scene_num) + '.txt'
    dqn_env = environment.Environment(testing_file=scene_name)
    #dqn_env.new_scene(group_num=0,scene_num=1)     





