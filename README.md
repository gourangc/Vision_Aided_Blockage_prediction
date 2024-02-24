# Computer Vision Aided Blockage Prediction in  Real-World Millimeter Wave Deployments
This is a python code package related to the following article:
Gouranga Charan and Ahmed Alkhateeb, "[Computer Vision Aided Blockage Prediction in  Real-World Millimeter Wave Deployments]

# Abstract of the Article
This paper provides the first real-world evaluation of using visual (RGB camera) data and machine learning for proactively predicting millimeter wave (mmWave) dynamic link blockages before they happen. Proactively predicting line-of-sight (LOS) link blockages enables mmWave/sub-THz networks to make proactive network management decisions, such as proactive beam switching and hand-off) before a link failure happens. This can significantly  enhance the network reliability and latency  while efficiently utilizing the wireless resources. To evaluate this gain in reality, this paper (i) develops a computer vision based solution that processes the visual data captured by a camera installed at the infrastructure node and (ii) studies the feasibility of the proposed solution based on the large-scale real-world dataset, DeepSense 6G, that comprises multi-modal sensing and communication data. Based on the adopted  real-world dataset, the developed solution achieves ~90% accuracy in predicting blockages happening within the future 0.1s and ~80% for blockages happening within 1s, which highlights a promising solution for  mmWave/sub-THz communication networks.

# Code Package Content 
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenarios 17-22 of DeepSense6G dataset.


# Image Enhancement
For the image enhancement step, we adopt the MIRNet network as proposed in the paper by Zamir, Syed Waqas, et al., "Learning enriched features for fast image restoration and enhancement." (IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, 2022, pp. 1934-1948). For further details and official implementation, please refer to the [MIRNet GitHub Repository](https://github.com/swz30/MIRNet).

If you utilize this approach in your work, please ensure to cite the paper appropriately


# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> Gouranga Charan and Ahmed Alkhateeb, "[Computer Vision Aided Blockage Prediction in Real-World Millimeter Wave Deployments](https://ieeexplore.ieee.org/abstract/document/10008524),", in 2022 IEEE Globecom Workshops (GC Wkshps), 2022, pp. 1711-1716.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net

