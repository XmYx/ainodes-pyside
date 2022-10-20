import sys

from PySide6.Qt3DRender import Qt3DRender
from PySide6.QtCore import Property, QObject, QPropertyAnimation, Signal, QUrl, QDir
from PySide6.QtGui import QGuiApplication, QMatrix4x4, QQuaternion, QVector3D
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras




class OrbitTransformController(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self._target = None
        self._matrix = QMatrix4x4()
        self._radius = 1
        self._angle = 0

    def setTarget(self, t):
        self._target = t

    def getTarget(self):
        return self._target

    def setRadius(self, radius):
        if self._radius != radius:
            self._radius = radius
            self.updateMatrix()
            self.radiusChanged.emit()

    def getRadius(self):
        return self._radius

    def setAngle(self, angle):
        if self._angle != angle:
            self._angle = angle
            self.updateMatrix()
            self.angleChanged.emit()

    def getAngle(self):
        return self._angle

    def updateMatrix(self):
        self._matrix.setToIdentity()
        self._matrix.rotate(self._angle, QVector3D(0, 1, 0))
        self._matrix.translate(self._radius, 0, 0)
        if self._target is not None:
            self._target.setMatrix(self._matrix)

    angleChanged = Signal()
    radiusChanged = Signal()
    angle = Property(float, getAngle, setAngle, notify=angleChanged)
    radius = Property(float, getRadius, setRadius, notify=radiusChanged)



class Window(Qt3DExtras.Qt3DWindow):
    def __init__(self):
        super().__init__()
        self.x_dist = 0
        self.y_dist = 0
        self.last_x = -1
        self.last_y = -1

        # Camera
        self.camera().lens().setPerspectiveProjection(10, 16 / 9, 0.1, 100)
        self.camera().setPosition(QVector3D(0, 0, 2))
        self.camera().setViewCenter(QVector3D(0, 0, 0))

        # For camera controls
        self.createScene()
        #self.camController = Qt3DExtras.QOrbitCameraController(self.rootEntity)
        #self.camController.setLinearSpeed(50)
        #self.camController.setLookSpeed(180)
        #self.camController.setCamera(self.camera())

        self.setRootEntity(self.rootEntity)
        print('mesh status: ')
        print(self.mesh.status())
        print(self.mesh.source())
        print(QDir.currentPath())


    def mouseMoveEvent(self, arg__1):
        if  arg__1.buttons():
            if self.last_x == -1:
                self.last_x = arg__1.x()
            else:
                self.x_dist = self.last_x - arg__1.x()
                self.last_x = arg__1.x()

            if self.last_y == -1:
                self.last_y = arg__1.y()
            else:
                self.y_dist = self.last_y - arg__1.y()
                self.last_y = arg__1.y()

            print(self.cameraTransform.rotationX())
            self.cameraTransform.setRotationX(self.cameraTransform.rotationX()-(self.y_dist*3))
            self.cameraTransform.setRotationY(self.cameraTransform.rotationY()-(self.x_dist*3))

            #self.cameraTransform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 0), 0))

            print(arg__1.x(), ' ',arg__1.y(), ' ', self.last_x, ' ',self.last_y, ' ', self.x_dist, ' ',self.y_dist, ' ', arg__1.buttons())

    def createScene(self):
        # Root entity
        self.rootEntity = Qt3DCore.QEntity()

        # Material
        self.material  = Qt3DExtras.QPhongMaterial(self.rootEntity)

        # Chest Entity
        self.camEntity = Qt3DCore.QEntity(self.rootEntity)
        self.mesh = Qt3DRender.QSceneLoader(self.rootEntity)
        file = QUrl.fromLocalFile("camera_model/cam.obj")
        self.mesh.setSource(file)

        self.cameraTransform = Qt3DCore.QTransform()
        self.cameraTransform.setScale3D(QVector3D(1, 1, 1))
        self.cameraTransform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(0, 0, 0), 0))

        self.camEntity.addComponent(self.mesh)
        self.camEntity.addComponent(self.material)
        self.camEntity.addComponent(self.cameraTransform)


        """
         # Material
        self.material = Qt3DExtras.QPhongMaterial(self.rootEntity)

        # Torus
        self.torusEntity = Qt3DCore.QEntity(self.rootEntity)
        self.torusMesh = Qt3DExtras.QTorusMesh()
        self.torusMesh.setRadius(5)
        self.torusMesh.setMinorRadius(1)
        self.torusMesh.setRings(100)
        self.torusMesh.setSlices(20)

        self.torusTransform = Qt3DCore.QTransform()
        self.torusTransform.setScale3D(QVector3D(1.5, 1, 0.5))
        self.torusTransform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 45))

        self.torusEntity.addComponent(self.torusMesh)
        self.torusEntity.addComponent(self.torusTransform)
        self.torusEntity.addComponent(self.material)

        # Sphere
        self.sphereEntity = Qt3DCore.QEntity(self.rootEntity)
        self.sphereMesh = Qt3DExtras.QSphereMesh()
        self.sphereMesh.setRadius(3)

        self.sphereTransform = Qt3DCore.QTransform()
        self.controller = OrbitTransformController(self.sphereTransform)
        self.controller.setTarget(self.sphereTransform)
        self.controller.setRadius(20)

        self.sphereRotateTransformAnimation = QPropertyAnimation(self.sphereTransform)
        self.sphereRotateTransformAnimation.setTargetObject(self.controller)
        self.sphereRotateTransformAnimation.setPropertyName(b"angle")
        self.sphereRotateTransformAnimation.setStartValue(0)
        self.sphereRotateTransformAnimation.setEndValue(360)
        self.sphereRotateTransformAnimation.setDuration(10000)
        self.sphereRotateTransformAnimation.setLoopCount(-1)
        self.sphereRotateTransformAnimation.start()

        self.sphereEntity.addComponent(self.sphereMesh)
        self.sphereEntity.addComponent(self.sphereTransform)
        self.sphereEntity.addComponent(self.material)
        
        """
