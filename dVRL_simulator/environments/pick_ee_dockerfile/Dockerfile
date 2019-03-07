FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04

WORKDIR /app

#Install a few essentials
RUN apt-get -y update && apt-get -y install libglib2.0-0 \ 
libpng12-0 libqscintilla2-dev liblua5.1-0-dev libqt5serialport5-dev libqt5opengl5-dev mesa-utils libgl1-mesa-glx wget xvfb

#LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV PATH /usr/local/nvidia/bin:${PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
  ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
  ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

#Download and uncompress v-rep
RUN mkdir -p /app/ \
&& wget -SL coppeliarobotics.com/files/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz \
&& tar -xzf /app/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz -C /app/ \
&& mv /app/V-REP_PRO_EDU_V3_5_0_Linux /app/V-REP \
&& rm /app/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz

#Expose the ports used in the scene
EXPOSE 80 
EXPOSE 19996 
EXPOSE 19997 
EXPOSE 19998 
EXPOSE 19999

#Copy in our scene
#ADD dvrk-vrep/V-REP_scenes/dVRK-oneArm-reach.ttt /app/scene.ttt
ADD dVRK-oneArm-pick.ttt /app/scene.ttt


#Hack used to reset LIBRARY PATH. In nvidia/opengl: https://gitlab.com/nvidia/opengl/blob/ubuntu16.04/base/Dockerfile
#Look at the last line :(
ENV LD_LIBRARY_PATH /usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu

#Run vrep with the scene and start simulation immediately
CMD /app/V-REP/vrep.sh -s -q /app/scene.ttt
