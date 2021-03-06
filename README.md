# Video-Advertisement-Classification
Video Advertisement Classification
Automatic identification of commercial blocks in news videos finds a lot of applications in the domain of television broadcast analysis and monitoring. Commercials occupy almost 40-60% of total air time. Manual segmentation of commercials from thousands of TV news channels is time consuming, and economically infeasible hence prompts the need for machine learning based method. 

Classifying TV News commercials is a semantic video classification problem. TV News commercials on particular news channel are combinations of video shots uniquely characterized by audio-visual presentation. Hence various audio visual features extracted from video shots are widely used for TV commercial classification. 

A version of this data set was used in the paper:

A. Vyas, R. Kannao, V. Bhargava and P. Guha, "Commercial Block Detection in Broadcast News Videos," in Proc. Ninth Indian Conference on Computer Vision, Graphics and Image Processing, Dec. 2014.

and portions of this description are taken from that source and from the description of the data set in the UCI Machine Learning Repository.

This data set contains data from Indian News channels.  These channels do not follow any particular news presentation format, have large variability and dynamic nature presenting a challenging machine learning problem. This data set contains instances from two news channels. Videos are recorded at resolution of 720 X 576 at 25 fps using a DVR and set top box. The feature file preserves the order of occurrence of shots.

Video shots are used as unit for generating instances. Broadcast News videos are segmented into video shots using RGB Colour Histogram matching Between consecutive video frames. From each video shot we have extracted 7 Audio (viz. Short term energy, zero crossing rate, Spectral Centroid, spectral Flux, spectral Roll off frequency, fundamental frequency and MFCC Bag of Audio Words) and 5 visual Features (viz. Video shot length, Screen Text Distribution, Motion Distribution, Frame Difference Distribution, Edge Change Ratio) from each video shot. Details of each extracted feature are as follows. 

In general to attract viewer's attention TV commercials have higher audio amplitude, appropriate background music (comparatively higher frequencies) as well as sharp transitions from one music to other or music to speech etc. We try to capture these properties by using low level audio features -- Short Time Energy (STE) , Zero Crossing Rate (ZCR), Spectral Centroid, Spectral Flux, Spectral Roll-Off Frequency and Fundamental Frequency. All of these short term audio features are calculated with audio frame size of 20 msec at 8000Hz sampling Frequency. The Mean and standard deviation of all audio feature values are calculated over the shot, generating a 2D vector for each feature. 

The MFCC Bag of Audio Words have been successfully used in several existing speech/audio processing applications. This motivated us to compute the MFCC coefficients along with Delta and Delta-Delta Cepstrum from 150 hours of audio tracks. These coefficients are clustered into 4000 groups which form the Audio words. Each shot is then represented as a 4000 Dimensional Bag of Audio Words by forming the normalized histograms of the MFCC's extracted from 20 ms windows with overlap of 10 ms in the shots. 

Commercial video shots are usually short in length, fast visual transitions with peculiar placement of overlaid text bands. Video Shot Length is directly used as one of the feature. Placement of overlaid text bands is represented by 15 dimensional overlaid Text Distribution. To calculate Text Distribution feature, video frame is divided into a grid of size 5 X 3(15 grid blocks). The text distribution feature is obtained by averaging the fraction of text area present in a grid block over all frames of the shot. Motion Distribution, Frame Change Distribution and Edge Change Ratio captures the dynamic nature of the commercial shots. 

Motion Distribution is obtained by first computing dense optical flow (Horn-Schunk formulation) followed by construction of a distribution of flow magnitudes over the entire shot with 40 uniformly divided bins in range of [0, 40]. Sudden changes in pixel intensities are grasped by Frame Difference Distribution. Such changes are not registered by optical flow. Thus, Frame Difference Distribution is also computed along with flow magnitude distributions. We obtain the frame difference by averaging absolute frame difference in each of 3 color channels and the distribution is constructed with 32 bins in the range of [0, 255] . Edge Change Ratio Captures the motion of edges between consecutive frames and is defined as ratio of displaced edge pixels to the total number of edge pixels in a frame. We calculate the mean and variance of the ECR over the entire shot. 

The labels are - +1/-1 (Commercials / Non-Commercials) 

The features are:

1 Shot Length

2-3 Motion Distribution(Mean and Variance)

4-5 Frame Difference Distribution (Mean and Variance)

6-7 Short time energy (Mean and Variance)

8-9 ZCR (Mean and Variance)

10-11 Spectral Centroid (Mean and Variance)

12-13 Spectral Roll off (Mean and Variance)

14-15 Spectral Flux (Mean and Variance)

16-17 Fundamental Frequency (Mean and Variance)

18-58 Motion Distribution (40 bins)

59-91 Frame Difference Distribution (32 bins)

92-122 Text area distribution (15 bins Mean and 15 bins for variance)

123-124 Edge change Ratio (Mean and Variance) 
