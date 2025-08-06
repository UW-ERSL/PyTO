from solidworks_interface import SolidWorksInterface
import matplotlib.pyplot as plt
import numpy as np
input("Is SolidWorks open with a TwoHolesTang?\nPress Enter to continue...")
solidWorks = SolidWorksInterface()
dimName1 = "D7@Sketch1"
p1Default = 0.034

dimName2 = "D8@Sketch1"
p2Default = 0.019

success =solidWorks.setDimensions([dimName1,dimName2], [p1Default,p2Default])
edgeCount0 = solidWorks.getEdgeCount()
faceCount0 = solidWorks.getFaceCount()
print("edgeCount0: ", edgeCount0)
print("faceCount0: ", faceCount0)   
p1Min = 0.010
p1Max = 0.090
p2Min = 0.025
p2Max = 0.060
N = 200
# Generate evenly spaced points for p1 and p2
p1Range = np.random.uniform(p1Min, p1Max, N)
p2Range = np.random.uniform(p2Min, p2Max, N)
nSuccess = 0
results = []
for i in range(N):
    print(i)
    p1 = p1Range[i]
    p2 = p2Range[i]

    # Set the dimensions
    success = solidWorks.setDimensions([dimName1,dimName2], [p1,p2])
    if not success:
        results.append((p1, p2, False))
        continue
    else:
        edgeCount = solidWorks.getEdgeCount()
        faceCount = solidWorks.getFaceCount()
        if edgeCount == edgeCount0 and faceCount == faceCount0:
            results.append((p1, p2, True))
            nSuccess += 1
            print("nSuccess: ", nSuccess)
        else:
            results.append((p1, p2, False))
        
 
solidWorks.setDimensions([dimName1,dimName2],[p1Default,p2Default])
# Separate the points based on success/failure
success_points = [(p1, p2) for p1, p2, success in results if success]
failure_points = [(p1, p2) for p1, p2, success in results if not success]

# Create scatter plot
plt.figure(figsize=(8, 6))
if success_points:
    p1_success, p2_success = zip(*success_points)
    plt.scatter(p1_success, p2_success, c='green', label='Success')
if failure_points:
    p1_fail, p2_fail = zip(*failure_points)
    plt.scatter(p1_fail, p2_fail, c='red', label='Failure')
plt.axis('equal')
plt.xlabel('p1')
plt.ylabel('p2')
plt.title('Parameter Space Results')
plt.legend()
plt.grid(True)
plt.show()