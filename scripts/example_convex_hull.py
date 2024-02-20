""" Create a four cube meshes  """

import SVMTK as svm
from pathlib import Path


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
   # Create Convex hull 
   convex_hull = surf.convex_hull()  
   # Remesh surface 
   convex_hull.isotropic_remeshing(.5,5,False)
   # Adjust vertices in normal direction 
   convex_hull.adjust_boundary(5.)
   # Remesh surface/ smooth the surface 
   convex_hull.isotropic_remeshing(0.5,5,False)   
   convex_hull.save("convex_hull.stl") 
   

   print("Start meshing convex hull") 
   maker = svm.Domain([convex_hull,surf])
   maker.create_mesh(64) 
   maker.save(Z.out)   

   print("Finish ",__file__)        
   
