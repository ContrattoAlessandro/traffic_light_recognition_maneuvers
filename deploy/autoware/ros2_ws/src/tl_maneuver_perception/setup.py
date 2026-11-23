from glob import glob

from setuptools import find_packages, setup

name = "tl_maneuver_perception"
setup(
    name=name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + name]),
        ("share/" + name, ["package.xml"]),
        ("share/" + name + "/launch", glob("launch/*.py")),
        ("share/" + name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Alessandro Contratto",
    maintainer_email="alexandro.contratto@gmail.com",
    description="RF-DETR model R traffic light maneuver perception for Autoware",
    license="Apache-2.0",
    entry_points={"console_scripts": ["tl_maneuver_node = tl_maneuver_perception.node:main"]},
)
