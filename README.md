# Brain MRI Pulse Sequences Classification

## **About Project**
This is official code of our package [brainmri_ps](https://pypi.org/project/brainmri-ps/). We provide a machine learning based tool to automatically classify Brain MRI series into different pulse sequence types:
- FLAIR
- T1C
- T2
- ADC
- DWI
- TOF
- OTHER

## **Installation**
Install via pip:
```
pip install brainmri_ps
```

## **Usage**

Load pretrained models:
```
from brainmri_ps import PulseSequenceClassifier
classifier = PulseSequenceClassifier("mobilenet_v2").from_pretrained()
```
|*Name*|*Input Resolution*|*#Params (M)*|*MACs (G)*|*Test Accuracy*|*Pretrained*|
|:----:|:----------------:|:-----------:|:--------:|:-------------:|:----------:|
|[MobileNet V2](https://arxiv.org/abs/1801.04381)|256|2.23|0.42|100.0|✓|

Example - predict from a study:
```
In  : classifier.predict_study("*/1.2.840.113619.6.388.6361536015762131135133837693432843617")
Out :
{
    "1.2.840.113619.2.5.1821162425615901145251590114525252000":   "ADC", 
    "1.2.840.113619.2.388.57473.14165493.12954.1590103413.819":    "T2", 
    "1.2.840.113619.2.388.57473.14165493.12954.1590103413.822":   "DWI", 
    "1.2.840.113619.2.388.57473.14165493.12954.1590103413.823":   "T1C", 
    "1.2.840.113619.2.388.57473.14165493.12954.1590103413.821": "FLAIR"
}
```
Function `predict_study` does the following steps:
- Read all dicom files in a study folder and group them into series by SeriesInstanceUID field
- Determine the orientation plane (axial, sagittal, coronal) of the series by using the ImageOrientationPatient field
- Predict and return the pulse sequence types of axial series (ignore the non-axial ones)

## **Contact**
Issues should be raised directly in the repository. For further support please email us at:
- lhkhiem28@gmail.com
- vantuan5644@gmail.com