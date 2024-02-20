""" Create a four cube meshes  """

import SVMTK as svm
from pathlib import Path



def bounding_radius(surf):
    # The surface span in dir 0:x,1:y,2:z
    xmin,xmax = surf.span(0)
    ymin,ymax = surf.span(1)
    zmin,zmax = surf.span(2)     
    
    return 0.5*((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
       

if __name__ == "__main__":
   print("Start ",__file__)   
   
   import argparse
   import os 

   parser = argparse.ArgumentParser(prog='write to excal.py')
   parser.add_argument("--surf")
   parser.add_argument("--out")
   Z = parser.parse_args()  
   # Load Surface
   surf = svm.Surface(Z.surf)
   
   # Get centeroid 
   centeroid = surf.centeroid()
   # Get bounding radius 
   radius = bounding_radius(surf)

   print( centeroid, radius )
   print("Making bounding sphere") 
   # Make sphere   
   sphere = svm.Surface()   
   sphere.make_sphere(centeroid, radius, 2.0 )
   sphere.save("sphere.stl") 
   

   print("Start meshing") 
   maker = svm.Domain([sphere,surf])
   maker.create_mesh(24) 
   maker.save(Z.out)   

   print("Finish ",__file__)        
   
