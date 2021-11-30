from setuptools import setup, find_packages

setup(
    name='networks_minimal',
    version='0.1',
    packages=find_packages(),
    url='https://https://github.com/gsiddhant/networks_minimal',
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=3.6.0',
    install_requires=[
        'torch',
        'numpy',
    ]
)
