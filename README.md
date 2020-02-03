# RL Environments for the da Vinci Surgical System

Arxiv paper: https://arxiv.org/abs/1903.02090

Video link : https://www.youtube.com/watch?v=xu4sqrO_2AY

System set up, note that only linux is supported:
1) Requires Docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/
2) Requires NVIDIA Container Runtime for Docker: https://github.com/NVIDIA/nvidia-docker
2) Enable GUI for Docker containers: http://wiki.ros.org/docker/Tutorials/GUI
3) Run bash script to build Docker Images: bash build_dockers.sh
4) Python packages: pip install transforms3d docker gym matplotlib

To speed up the simulation during training, V-REP can be launched within the docker in hidden mode. 

To turn this on modify the last line in dVRL_simulator/environments/<reach/pick>_ee_dockerfile/Dockerfile. Add the "-h" flag in the final line: 

	CMD /app/V-REP/vrep.sh -h -s -q /app/scene.ttt

