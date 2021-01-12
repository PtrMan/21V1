class Scene(object):
    def __init__(self):
        self.backgroundColor = [0.0, 0.0, 0.0]

        self.cameraPos = [0.0, 0.0, 0.0]
        self.lookAt = [0.0, 0.0, 1.0]

        self.lightPos = [0.0, 0.0, 0.0]
        
        self.objs = [] # objects in the scene, class Obj

        self.boxCenters = [[-0.1, 0.0, -2.8]] # legacy
        self.spheres = [([0.0, 0.0, -2.8], 0.03)] # legacy

        self.enBox = True # legacy



class Obj(object):
    # /param type_ name of type of object, "s" for sphere, "b" for box
    # /param extend can be single value for radius or 3d vector for box
    def __init__(self, type_, center, extend):
        self.type_ = type_
        self.center = center
        self.extend = extend
        self.isTextured = False # is a standard texture applied?

# generate povray scene, write it to file, render file
# write png to local file
def renderScene(scene, displaySize):
    sceneTemplate = """
    #include "colors.inc"
    #include "textures.inc"

background { color rgb BGCOLOR }

camera {
  location <CAMERAPOS>
  look_at <LOOKAT>
}

OBJS

light_source { <LIGHTPOS> color White}
""".replace("BGCOLOR", f'<{scene.backgroundColor[0]},{scene.backgroundColor[1]},{scene.backgroundColor[2]}>')
    sceneContent = sceneTemplate[:]
    sceneContent = sceneContent.replace("CAMERAPOS", str(scene.cameraPos[0])+","+str(scene.cameraPos[1])+","+str(scene.cameraPos[2]))

    sceneContent = sceneContent.replace("LIGHTPOS",  str(scene.lightPos[0])+","+str(scene.lightPos[1])+","+str(scene.lightPos[2]))

    sceneContent = sceneContent.replace("LOOKAT",  str(scene.lookAt[0])+","+str(scene.lookAt[1])+","+str(scene.lookAt[2]))


    objsText = ""

    for iSphereCenter, iSphereR in scene.spheres:
        spherePosAsStr = f'<{iSphereCenter[0]},{iSphereCenter[1]},{iSphereCenter[2]}>, {iSphereR}'

        objsText += """
sphere {
  POSS
  texture {
    pigment { color Yellow }
  }
}""".replace("POSS", spherePosAsStr) # replace position with src-code of position

    for iBoxCenter in scene.boxCenters:
        boxextend = [0.08, 0.08, 0.08]

        # positions of edges of box as string
        edgePointsAsStr = f"<{iBoxCenter[0]-boxextend[0]/2.0},{iBoxCenter[1]-boxextend[1]/2.0},{iBoxCenter[2]-boxextend[2]/2.0}>, <{iBoxCenter[0]+boxextend[0]/2.0},{iBoxCenter[1]+boxextend[1]/2.0},{iBoxCenter[2]+boxextend[2]/2.0}>"
        
        objsText += """
box {
  POSS
  texture {
    pigment { color Yellow }
  }
}
""".replace("POSS", edgePointsAsStr) # replace position with src-code of position


    sceneContent = sceneContent.replace("OBJS", objsText)

    # append objects
    for iObj in scene.objs:
        povType = None
        povPos = None

        if iObj.type_ == "s": # is sphere?
            povType = "sphere"
            povPos = f'<{iObj.center[0]},{iObj.center[1]},{iObj.center[2]}>, {iObj.extend}'
        elif iObj.type_ == "b": # is box?
            povType = "box"
            povPos = f"<{iObj.center[0]-iObj.extend[0]/2.0},{iObj.center[1]-iObj.extend[1]/2.0},{iObj.center[2]-iObj.extend[2]/2.0}>, <{iObj.center[0]+iObj.extend[0]/2.0},{iObj.center[1]+iObj.extend[1]/2.0},{iObj.center[2]+iObj.extend[2]/2.0}>"
        else:
            raise Exception(f'Invalid type "{iObj.type_}"')
        
        

        povTexture = "texture { pigment { color Yellow } }"
        if iObj.isTextured:
            povTexture = "texture { Blue_Sky2 }" # UNTESTED

        sceneContent += f'{povType} {{ {povPos} {povTexture} }}  \n'


    f = open("TEMPScene.pov", 'w')
    f.write(sceneContent)
    f.close()

    import subprocess
    subprocess.call(["povray", "TEMPScene.pov", "+W"+str(displaySize[0]), "+H"+str(displaySize[1])], stderr=subprocess.PIPE)
