import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='pyaiutils',  
     version='0.1',
     scripts=['pyaiutils'] ,
     author="Prabhat Kumar de Oliveira, Guilherme Silva da Cunha, Erick Sperandio",
     author_email="bartkoliveira@gmail.com, guiscunha@gmail.com, erick.sperandio@fieb.org.br",
     description="Description",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Bart-BK/pyaiutils",
     python_requires='>=3.6',
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: Apache License",
         "Operating System :: OS Independent",
     ],
 )