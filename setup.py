from setuptools import setup

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='brainmri_ps',
    version='1.0.2',
    packages=['brainmri_ps'],
    url='',
    license='LICENSE.txt',
    author='lhkhiem, tuantran',
    author_email='',
    description='Automatically classify Brain MRI series by pulse sequence types: FLAIR, T1C, T2, ADC, DWI, TOF and OTHER',
    install_requires=install_requires,
)