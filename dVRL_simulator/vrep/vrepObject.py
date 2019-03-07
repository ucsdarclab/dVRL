# This can be used if the vrep connection is already made adn clientID is passed
# V-REP must be running in synchronuous mode to use this.
# So the trigger to continue simulation must be done outside of this class
# To ensure data is sync'ed properly. Use the following two commands outside of this class:
#		vrep.simxSynchronousTrigger(clientID)
#		vrep.simxGetPingTime(cliendID)

try:
	import dVRL_simulator.vrep.vrep as vrep
except:
	print ('--------------------------------------------------------------')
	print ('"vrep.py" could not be imported. This means very probably that')
	print ('either "vrep.py" or the remoteApi library could not be found.')
	print ('Make sure both are in the same folder as this file,')
	print ('or appropriately adjust the file "vrep.py"')
	print ('--------------------------------------------------------------')
	print ('')

import numpy as np
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler


class vrepObject:
	def __init__(self, clientID):
		self.clientID = clientID

	#Wrapper for simxSetIntegerSignal
	def setIntegerSignal(self, integerString, value, ignoreError = False):
		res = vrep.simxSetIntegerSignal(self.clientID, integerString, value,  vrep.simx_opmode_oneshot)
		
		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set integer signal {}'.format(integerString))
			print(res)

	#Wrapper for simxGetObjectParent
	def getParent(self, handle_obj, ignoreError = False, initialize = False):

		if initialize:
			vrep.simxGetObjectParent(self.clientID, handle_obj, vrep.simx_opmode_streaming)

		res, out = vrep.simxGetObjectParent(self.clientID, handle_obj,  vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get parent from object')
			print(res)

		return out

	#Wrapper for simxSetObjectParent
	def setParent(self, handle_obj, handle_parent, retainPosition, ignoreError = False):
		res = vrep.simxSetObjectParent(self.clientID, handle_obj, handle_parent, retainPosition, vrep.simx_opmode_oneshot)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set parent for object')
			print(res)


	#Wrapper for simxSetBooleanParameter
	# Look here to see what booleans you can modify: http://www.coppeliarobotics.com/helpFiles/en/apiConstants.htm#booleanParameters
	def setBooleanParameter(self, paramIdentifier, paramValue, ignoreError = False):
		res = vrep.simxSetBooleanParameter(self.clientID, paramIdentifier, paramValue, vrep.simx_opmode_blocking)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set {}'.format(paramIdentifier))
			print(res)

	#Wrapper for simxGetBooleanParameter
	def getBooleanParameter(self, paramIdentifier, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxGetBooleanParameter(self.clientID, paramIdentifier, vrep.simx_opmode_streaming)

		res, out = vrep.simxGetBooleanParameter(self.clientID, paramIdentifier,  vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get {}'.format(paramIdentifier))
			print(res)

		return out

	#Wrapper for simxGetObjectHandle
	def getHandle(self, name, ignoreError = False):
		res, out_handle = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to find {}'.format(name))
			print(res)

		return out_handle

	#Wrapper for simxGetVisionSensorImage
	def getVisionSensorImage(self, handle, rgb = True, ignoreError = False, initialize = False):
		if rgb:
			b = 0
		else:
			b = 1

		if initialize:
			vrep.simxGetVisionSensorImage(self.clientID, handle, b, vrep.simx_opmode_streaming)

		res, image_resolution, image_data = vrep.simxGetVisionSensorImage(self.clientID, handle, b, vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get image')
			print(res)

		return image_data, image_resolution

	#Wrapper for simxGetCollisionHandle
	def getCollisionHandle(self, name, ignoreError = False):
		res, out_handle = vrep.simxGetCollisionHandle(self.clientID, name, vrep.simx_opmode_blocking)

		if res != vrep.simx_return_ok and not ignoreError:
			print('Failed to find {}'.format(name))
			print(res)

		return out_handle

	#Wrapper for simxReadCollision
	def checkCollision(self, handle, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxReadCollision(self.clientID, handle, vrep.simx_opmode_streaming)

		res, collision = vrep.simxReadCollision(self.clientID, handle, vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to check collision')
			print(res)

		return collision

	#Wrapper for simxGetJointPosition
	def getJointPosition(self, handle, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxGetJointPosition(self.clientID, handle,  vrep.simx_opmode_streaming)

		res, out_pos = vrep.simxGetJointPosition(self.clientID, handle,  vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get joint angle')
			print(res)

		return out_pos

	#Wrapper to get joint velocity
	def getJointVelocity(self, handle, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxGetObjectFloatParameter(self.clientID, handle, 2012, vrep.simx_opmode_streaming)

		res,velocity=vrep.simxGetObjectFloatParameter(self.clientID, handle, 2012, vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get joint velocity')
			print(res)

		return velocity

	#Wrapper for simxSetJointPosition 
	def setJointPosition(self, handle, position, ignoreError = False, initialize = False):
		res = vrep.simxSetJointPosition(self.clientID, handle, position, vrep.simx_opmode_oneshot)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set joint angle')
			print(res)

	#Wrapper for simxGetObjectPosition and simxGetObjectQuaternion
	def getPoseAtHandle(self, targetHandle, refHandle, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxGetObjectPosition(  self.clientID, targetHandle, refHandle, vrep.simx_opmode_streaming)
			vrep.simxGetObjectQuaternion(self.clientID, targetHandle, refHandle, vrep.simx_opmode_streaming)

		res1, pos  = vrep.simxGetObjectPosition(  self.clientID, targetHandle, refHandle, vrep.simx_opmode_buffer)
		res2, quat = vrep.simxGetObjectQuaternion(self.clientID, targetHandle, refHandle, vrep.simx_opmode_buffer)

		if res1!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get position')
			print(res1)
		if res2!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get orientation')
			print(res2)

		return np.array(pos), np.array(quat)

	#Wrapper for simxGetObjectVelocity
	def getVelocityAtHandle(self, targetHandle, ignoreError = False, initialize = False):
		if initialize:
			vrep.simxGetObjectVelocity(self.clientID, targetHandle, vrep.simx_opmode_streaming)

		res, l_vel, r_vel = vrep.simxGetObjectVelocity(self.clientID, targetHandle, vrep.simx_opmode_buffer)

		if res!=vrep.simx_return_ok and not ignoreError:
			print('Failed to get velocity')
			print(res)

		return np.array(l_vel), np.array(r_vel)

	#Wrapper for simxSetObjectPosition and simxSetObjectQuaternion
	def setPoseAtHandle(self, targetHandle, refHandle, pos, quat, ignoreError = False):
		res1 = vrep.simxSetObjectPosition(  self.clientID, targetHandle, refHandle, pos,  vrep.simx_opmode_oneshot)
		res2 = vrep.simxSetObjectQuaternion(self.clientID, targetHandle, refHandle, quat, vrep.simx_opmode_oneshot)

		if res1!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set position')
			print(res1)
		if res2!=vrep.simx_return_ok and not ignoreError:
			print('Failed to set orientation')
			print(res2)


	def posquat2Matrix(self, pos, quat):
		T = np.eye(4)
		T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
		T[0:3, 3] = pos

		return np.array(T)

	def matrix2posquat(self,T):
		pos = T[0:3, 3]
		quat = quaternions.mat2quat(T[0:3, 0:3])
		quat = [quat[1], quat[2], quat[3], quat[0]]

		return np.array(pos), np.array(quat)


	#Everything is defined here as the same as V-rep
	#RPY is relative
	#Quaternion is (i, j, k, w)
	#These functions are added just to make life easier and reduce number of errors...
	def euler2Quat(self, rpy):
		q = euler.euler2quat(rpy[0], rpy[1], rpy[2],  axes = 'rxyz')
		return np.array([q[1], q[2], q[3], q[0]])

	def quat2Euler(self, quat):
		return np.array(euler.quat2euler([quat[3], quat[0], quat[1], quat[2]], axes = 'rxyz'))