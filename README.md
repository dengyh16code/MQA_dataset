# MQA_dataset
This is a dataset of MQA task, you can get the simulation scene in V-REP, 3D object models, complicated scenes and corresponding question-anwer pairs.
- Paper MQA: https://arxiv.org/abs/2003.04641
- Source code:https://github.com/dengyh16code/MQA_ICRA2021
  
First, you need to unzip data.zip, then you will get:
```
├── README.md                   help
├── environment.py              python file for vrep remote api
├── remoteApi.dll               vrep configure file(Windows)
├── remoteApi.so                vrep configure file(Linux)
├── scene.ttt                   verp simlation file
├── test.py                     demo to show a simulation scene of MQA task
├── vrep.py                     python file for vrep configure
├── vrepConst.py                python file for vrep configure
├── data
│   ├── encode_ques             encoding question file
│   ├── mesh                    3D object models can be used in Vrep
│   ├── ques                    question file
│   ├── test_cases              scenes file
│   ├── boundary_size.json      size of 3D object\
│   ├── box.txt                 working space of UR5 in simulation
│   └── vocab.json              vocabulary
```

## Environments

- [V-REP3.5.0](https://www.coppeliarobotics.com/previousVersions)


## Usage
### open the scene

We bulid our dataset in a simulation environment named [V-REP](http://coppeliarobotics.com/). First, you need to open the our simulation scene file [Simulation.ttt](./scene.ttt) in [V-REP](http://coppeliarobotics.com/). 

![image](https://github.com/dengyh16code/MQA_dataset/blob/main/simulation.png)

### Using the python remote api to load scene
Then you can use the python scipt [test.py](./test.py) to load different scene in our dataset, where group_num can be taken from 0 to 9 and scene_num can be taken from 0 to 9. Different group_num and scene_num represent loading different scenes.

```sh
    python test.py -group_num 1 -scene_num 0 
```

![image](https://github.com/dengyh16code/MQA_dataset/blob/main/group_1_scene_0.png)

More remote api function(such as getting camera data, controlling UR5 manipulator, load questions) can be found in [enviroment.py](./enviroment.py).

## Citation

If you feel it useful, please cite:
```bibtex
@article{deng2020mqa,
  title={MQA: Answering the Question via Robotic Manipulation},
  author={Yuhong, Deng, and Xiaofeng, Guo, and Naifu, Zhang and Di, Guo and Huaping, Liu, and Fuchun, Sun},
  journal={arXiv preprint arXiv:2003.04641},
  year={2020}
}
```

## License

MIT License.
