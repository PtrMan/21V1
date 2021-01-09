class Scene(object):
    def __init__(self):
        self.cameraPos = [0.0, 0.0, 0.0]
        self.lookAt = [0.0, 0.0, 1.0]

        self.lightPos = [0.0, 0.0, 0.0]

        self.boxCenters = [[-0.1, 0.0, -2.8]]

        self.enBox = True
        self.enSphere = True

# generate povray scene, write it to file, render file
# write png to local file
def renderScene(scene, displaySize):
    sceneTemplate = """
    #include "colors.inc"    

background { color Black }

camera {
  location <CAMERAPOS>
  look_at <LOOKAT>
}

OBJS

light_source { <LIGHTPOS> color White}
"""
    sceneContent = sceneTemplate[:]
    sceneContent = sceneContent.replace("CAMERAPOS", str(scene.cameraPos[0])+","+str(scene.cameraPos[1])+","+str(scene.cameraPos[2]))

    sceneContent = sceneContent.replace("LIGHTPOS",  str(scene.lightPos[0])+","+str(scene.lightPos[1])+","+str(scene.lightPos[2]))

    sceneContent = sceneContent.replace("LOOKAT",  str(scene.lookAt[0])+","+str(scene.lookAt[1])+","+str(scene.lookAt[2]))


    objsText = ""

    if scene.enSphere:
        objsText += """
sphere {
  <0.0, 0.0, -2.8>, 0.03
  texture {
    pigment { color Yellow }
  }
}"""

    if scene.enBox:
        boxcenter = scene.boxCenters[0] # hardcoded to take first box center
        boxextend = [0.08, 0.08, 0.08]

        # positions of edges of box as string
        #edgePointsAsStr = "<-0.7, -0.05, -2.9>, <-0.5, 0.05, -2.7>"

        edgePointsAsStr = f"<{boxcenter[0]-boxextend[0]/2.0},{boxcenter[1]-boxextend[1]/2.0},{boxcenter[2]-boxextend[2]/2.0}>, <{boxcenter[0]+boxextend[0]/2.0},{boxcenter[1]+boxextend[1]/2.0},{boxcenter[2]+boxextend[2]/2.0}>"
        
        objsText += """
box {
  POSS
  texture {
    pigment { color Yellow }
  }
}
""".replace("POSS", edgePointsAsStr) # replace position with src-code of position


    sceneContent = sceneContent.replace("OBJS", objsText)


    f = open("TEMPScene.pov", 'w')
    f.write(sceneContent)
    f.close()

    #import os
    #os.system("povray TEMPScene.pov"+" +W"+str(displaySize[0])+" +H"+str(displaySize[1]))

    import subprocess
    subprocess.call(["povray", "TEMPScene.pov", "+W"+str(displaySize[0]), "+H"+str(displaySize[1])], stderr=subprocess.PIPE)
