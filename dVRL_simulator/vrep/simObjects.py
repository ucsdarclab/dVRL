from dVRL_simulator.vrep.vrepObject import vrepObject
import numpy as np

class camera(vrepObject):
	def __init__(self, clientID, rgb = True):
		super(camera, self).__init__(clientID)
		self.camera_handle = self.getHandle('Vision_Sensor')
		self.rgb = rgb

		self.getVisionSensorImage(self.camera_handle, self.rgb, ignoreError = True, initialize = True)

	def getImage(self, ignoreError = False):
		data, resolution = self.getVisionSensorImage(self.camera_handle, self.rgb, ignoreError = ignoreError, 
													initialize = False)

		if self.rgb:
			return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])
		else:
			return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]])


class collisionCheck(vrepObject):
	def __init__(self, clientID, psm_number):
		super(collisionCheck, self).__init__(clientID)

		self.collision_TTs_TableTop = self.getCollisionHandle('PSM{}_TTs_Table'.format(psm_number))
		self.collision_TTd_TableTop = self.getCollisionHandle('PSM{}_TTd_Table'.format(psm_number))

		super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError = True, initialize = True)
		super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError = True, initialize = True)

	#Returns True if in collision and False if not in collision
	def checkCollision(self, ignoreError = False):
		c1 = super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError)
		c2 = super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError)

		return c1 or c2

class table(vrepObject):
	def __init__(self, clientID):
		super(table, self).__init__(clientID)
		self.table_top_handle = self.getHandle('customizableTable_tableTop')


	def getPose(self, relative_handle, ignoreError = False, initialize = False):
		return self.getPoseAtHandle(self.table_top_handle, relative_handle, ignoreError, initialize)


#Dummy is +0.001 on z-axis
#Dummy is the actual thing you set/get since it is on the top of the cylinder
class obj(vrepObject):
	def __init__(self, clientID):
		super(obj, self).__init__(clientID)

		self.obj_handle = self.getHandle('Object')
		self.dummy_handle = self.getHandle('Object_Dummy')

	def setPose(self, pos, quat, relative_handle, ignoreError = False):
		b_T_d = self.posquat2Matrix(pos, quat)
		d_T_o = np.array([[1, 0, 0, 0], [0,1,0,0], [0,0,1,0.001], [0,0,0,1]])
		pos, quat = self.matrix2posquat(np.dot(b_T_d,d_T_o))

		self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

	def getPose(self, relative_handle, ignoreError = False, initialize = False):
		return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)

	def getVel(self, ignoreError = False, initialize = False):
		return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)

	def removeGrasped(self, ignoreError = False):
		self.setParent(self.obj_handle, -1, True, ignoreError)

	def isGrasped(self, ignoreError = False, initialize = False):
		return not (-1 == self.getParent(self.obj_handle, ignoreError, initialize))

class target(vrepObject):
	def __init__(self,clientID, psm_number):
		super(target, self).__init__(clientID)

		self.target_handle = self.getHandle('Target_PSM{}'.format(psm_number))

		self.getPosition(-1, ignoreError = True, initialize = True)

	def setPosition(self, pos, relative_handle, ignoreError = False):
		self.setPoseAtHandle(self.target_handle, relative_handle, pos, [1,0,0,1], ignoreError)

	def getPosition(self, relative_handle, ignoreError = False, initialize = False):
		pos, _ = self.getPoseAtHandle(self.target_handle, relative_handle, ignoreError, initialize)
		return pos
