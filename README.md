# MQA_dataset
This is a dataset of MQA task,  you can get the simulation scene in V-rep,  3D object models, complicated scenes  and  corresponding question-anwer pairs.
Vrep version:v3.5.0
First, you need to unzip data.zip,then you will get 

├── README.md                   // help
├── environment.py              // python file for vrep remote api
├── remoteApi.dll               // vrep configure file(Windows)
├── remoteApi.so                // vrep configure file(Linux)
├── scene.ttt                   // verp simlation file
├── test.py                     // demo to show a simulation scene of MQA task
├── vrep.py                     // python file for vrep configure 
├── vrepConst.py                // python file for vrep configure 
├── data
│   ├── encode_ques             // encoding question file
│   ├── mesh                    // 3D object models can be used in Vrep
│   ├── ques                    // question file
│   ├── test_cases              // scenes file
│   ├── boundary_size.json      // size of 3D object
│   ├── box.txt                 // working space of UR5 in simulation
│   └── vocab.json              // vocabulary

