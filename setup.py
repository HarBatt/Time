from setuptools import setup, find_packages
import TSG
version = TSG.__version__

setup(
    name='tsgeneration',
    version= '0.1.0',
    description='Juniper Time Series Generation library',
    maintainer='Harshavardhan Battula, Aman Gaurav, Ajit Patankar',
    maintainer_email='hbattula@juniper.net, agaurav@juniper.net, apatankar@juniper.net',
    license='Juniper@2022',
    url='https://ssd-git.juniper.net/cap/5g_ai_ml',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy==1.19.5',
        'torch','torch-scatter', 
        'torch-geometric'
    ],
    extras_require={'plotting': ['matplotlib==3.2.1', 'jupyter']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest', 'pytest-cov'],
    include_package_data=False,
    zip_safe=True
)
