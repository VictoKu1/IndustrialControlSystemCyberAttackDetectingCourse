# Industrial Control System Power - Cyber Attacks Detection 

###### Ariel University, Israel || Semester A, 2022

## Links

#### Datasets:

https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

#### Website:

http://157.230.22.122:5079/

## Docker usage instructions:

1. Pip requirements for the Jupyter notebook and the UI:
```
    pip install -r requirements.txt
```

2. Build the docker image:
```
    docker build -t attack_detection_ui
```

3. Run the docker container:
```
    docker run -it  attack_detection_ui
```




## [Presentation](https://www.canva.com/design/DAFbkwt6KjQ/7im1RNlKjYkdZpg1dugivg/view?utm_content=DAFbkwt6KjQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[![images_01](https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/blob/master/Media/1.gif)](https://www.canva.com/design/DAFbkwt6KjQ/7im1RNlKjYkdZpg1dugivg/view?utm_content=DAFbkwt6KjQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)






## Datasets

* [2 Classes](https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/blob/master/Class/binaryAllNaturalPlusNormalVsAttacks) - The 37 event scenarios were grouped as either an attack (28 events) or normal operations (9 events). The data was drawn from 15 data sets which included thousands of individual samples of measurements throughout the power system for each event type.

* [3 Classes](https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/blob/master/Class/triple) - The 37 event scenarios were grouped into 3 classes: attack events (28 events), natural event (8 events) or “No events” (1 event).

* [Multi-class](https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/blob/master/Class/multiclass) - Each of the 37 event scenarios, which included attack events, natural events, and normal operations, was its own class and was predicted independently by the learners,



## Power System Datasets (Dataset 1)

Uttam Adhikari, Shengyi Pan, and Tommy Morris in collaboration with Raymond Borges and Justin Beaver of Oak Ridge National Laboratories (ORNL) have created 3 datasets which include measurements related to electric transmission system normal, disturbance, control, cyber attack behaviors. Measurements in the dataset include synchrophasor measurements and data logs from Snort, a simulated control panel, and relays.

[README Description](http://www.google.com/url?q=http%3A%2F%2Fwww.ece.uah.edu%2F~thm0009%2Ficsdatasets%2FPowerSystem_Dataset_README.pdf&sa=D&sntz=1&usg=AOvVaw3t-soxdA-27GPUSRG1JP_Q)

The power system datasets have been used for multiple works related to power system cyber-attack classification.

* [Pan, S., Morris, T., Adhikari, U., Developing a Hybrid Intrusion Detection System Using Data Mining for Power Systems, IEEE Transactions on Smart Grid. doi: 10.1109/TSG.2015.2409775](http://www.google.com/url?q=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D7063234%26isnumber%3D5446437&sa=D&sntz=1&usg=AOvVaw06Q-fkYHriTfgJYieCBnJc)

* [Pan, S., Morris, T., Adhikari, U., Classification of Disturbances and Cyber-attacks in Power Systems Using Heterogeneous Time-synchronized Data, IEEE Transactions on Industrial Informatics. doi: 10.1109/TII.2015.2420951](http://www.google.com/url?q=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D7081776%26isnumber%3D4389054&sa=D&sntz=1&usg=AOvVaw21tCmn-MAAmkUzCRpflyv_)

* [Pan, S., Morris, T., Adhikari, U., A Specification-based Intrusion Detection Framework for Cyber-physical Environment in Electric Power System, International Journal of Network Security (IJNS), Vol.17, No.2, PP.174-188, March 2015.](http://www.google.com/url?q=http%3A%2F%2Fijns.jalaxy.com.tw%2Fcontents%2Fijns-v17-n2%2Fijns-2015-v17-n2-p174-188.pdf&sa=D&sntz=1&usg=AOvVaw3qkk5GcOIxcgHQesgQjr5w)

* [Beaver, J., Borges, R., Buckner, M., Morris, T., Adhikari, U., Pan, S., Machine Learning for Power System Disturbance and Cyber-attack Discrimination, Proceedings of the 7th International Symposium on Resilient Control Systems, August 19-21,2014, Denver, CO, USA.](https://www.google.com/url?q=https%3A%2F%2Fdoi.org%2F10.1109%2FISRCS.2014.6900095&sa=D&sntz=1&usg=AOvVaw3fR5r_1bSnchlVhDlEXHXE)


## Additional Articles

1. [Industrial Control System Traffic Datasets For Intrusion Detection Research](https://link.springer.com/content/pdf/10.1007/978-3-662-45355-1_5.pdf)

2. [Cyber-Attack Detection for Industrial Control System Monitoring with Support Vector Machine Based on Communication Profile](https://www.researchgate.net/profile/Ichiro-Koshijima/publication/318127445_Cyber-Attack_Detection_for_Industrial_Control_System_Monitoring_with_Support_Vector_Machine_Based_on_Communication_Profile/links/59f477b50f7e9b553ebbdeb6/Cyber-Attack-Detection-for-Industrial-Control-System-Monitoring-with-Support-Vector-Machine-Based-on-Communication-Profile.pdf)

3. [Efficient Cyber Attack Detection in Industrial Control Systems Using Lightweight Neural Networks and PCA](https://arxiv.org/pdf/1907.01216)

4. [Measuring the Risk of Cyber Attack in Industrial Control Systems](https://dora.dmu.ac.uk/bitstream/handle/2086/13839/ewic_icscsr2016_paper12.pdf?sequence=1)

5. [An Ensemble Deep Learning-Based Cyber-Attack Detection in Industrial Control System](https://ieeexplore.ieee.org/iel7/6287639/8948470/09086038.pdf)



































