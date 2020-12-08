# supposed to simulate primitive robot in a virtual 2.5 environment


# generate povray scene, write it to file, render file
# write png to local file
def renderScene(displaySize):
    sceneTemplate = """
    #include "colors.inc"    

background { color Black }

camera {
  location <0, 2, -3>
  look_at <0, 1, 2>
}

sphere {
  <0, 1, 40>, 2
  texture {
    pigment { color Yellow }
  }
}

light_source { <2, 4, -3> color White}
"""
    sceneContent = sceneTemplate[:]

    f = open("TEMPScene.pov", 'w')
    f.write(sceneContent)
    f.close()

    import os
    os.system("povray TEMPScene.pov"+" +W"+str(displaySize[0])+" +H"+str(displaySize[1]))




def main():
    import pygame


    pygame.init()


    displaySize = (400, 300)


    gameDisplay = pygame.display.set_mode(displaySize)
    pygame.display.set_caption('ENV')

    while True: # "game"loop
        # iterate over the list of Event objects 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # deactivates the pygame library 
                pygame.quit() 
  
                # quit the program. 
                quit() 

        black = (0, 0, 0)
        gameDisplay.fill(black) # fill background


        renderScene(displaySize) # render scene with renderer

        image = pygame.image.load("TEMPScene.png")     
        gameDisplay.blit(image, (0, 0)) 

        # Draws the surface object to the screen
        pygame.display.update()

main()