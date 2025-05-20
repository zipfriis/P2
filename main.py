import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

dt = None

def Calc_dt(xAirSpeed: float, 
            yAirSpeed: float, 
            dx: float, dy: float) -> tuple:
    """
    Calculates the time step it takes for air to travel one cell
    Args:
        airspeed (float): The speed of the air intake
        dx (float):       The length of the volumes in the x direction
        dy (float):       -||- y direction
    Returns:
        tuple: Contains the amount of time in seconds it takes for the
               air to go over one cell in either the x[0] or y[1] direction.
    """
    return (dx/xAirSpeed, dy/yAirSpeed)

def convectingArrayBool(xDisc: dict, yDisc: dict) -> np.ndarray:
    
    matrixShape = (xDisc["plate"] + xDisc["mount"], yDisc["plate"] + yDisc["mount"])
    convectingArray = np.full(matrixShape,False)
    
    #Slicing where the plate starts and ends
    xStart = xDisc["mount"] // 2
    xEnd = -xStart if xDisc["mount"] > 0 else None
    yStart = yDisc["mount"] // 2
    yEnd = -yStart if yDisc["mount"] > 0 else None
    

    #Slicing what is not the mount
    convectingArray[xStart : xEnd, yStart : yEnd] = True
    
    return convectingArray.reshape((matrixShape[0] * matrixShape[1])) 

def distribute_discretizations(total_discretizations: int, 
                               plate_length: float, 
                               mount_length: float) -> dict:
    """
    Distributes the total number of discretizations between the plate and mount
    based on their lengths, ensuring the sum matches the total discretizations.

    Args:
        total_discretizations (int): Total number of discretizations to distribute.
        plate_length (float): Length of the plate.
        mount_length (float): Length of the mount.

    Returns:
        dict: A dictionary with the discretizations for the plate and mount.
    """
    total_length = plate_length + mount_length

    # Calculate proportions
    plate_proportion = plate_length / total_length

    # Calculate initial discretizations
    plate_discretizations = round(plate_proportion * total_discretizations)
    mount_discretizations = total_discretizations - plate_discretizations  # Ensure the sum matches

    return {
        "plate": plate_discretizations,
        "mount": mount_discretizations
    }

def constructKMatrix(xCondCoeff: float, yCondCoeff: float,
                     botConvCoeff: float, topConvCoeff: float,
                     matrix2dSize: int, plateDisc: dict,
                     xDiscretizations: dict, yDiscretizations: dict) -> csr_matrix:
    
    matrix3dSize = matrix2dSize * 3
    coeffMatrix = lil_matrix((matrix3dSize,matrix3dSize))
    
    #Conduction
    for row in range(matrix2dSize, 2*matrix2dSize):
        #dT/dx
        if row % plateDisc["x"] == 0:
            #Forward difference method
            coeffMatrix[row, row] += xCondCoeff
            coeffMatrix[row, row + 1] += -2 * xCondCoeff
            coeffMatrix[row, row + 2] += xCondCoeff
        elif row % plateDisc["x"] == plateDisc["x"] - 1:
            #Backward difference method
            coeffMatrix[row, row] += xCondCoeff
            coeffMatrix[row, row - 1] += -2 * xCondCoeff
            coeffMatrix[row, row - 2] += xCondCoeff
        else:
            #Centered difference method
            coeffMatrix[row, row - 1] += xCondCoeff
            coeffMatrix[row, row] += -2 * xCondCoeff
            coeffMatrix[row, row + 1] += xCondCoeff
            
        #dT/dy
        above = plateDisc["x"]
        below = -plateDisc["x"]
        plateRow = row - matrix2dSize
        if plateRow < plateDisc["x"]:
            #Forward difference method
            coeffMatrix[row, row] += yCondCoeff
            coeffMatrix[row, row + above] += -2 * yCondCoeff
            coeffMatrix[row, row + 2 * above] += yCondCoeff
        elif plateRow >= matrix2dSize - plateDisc["x"]:
            #Backward difference method
            coeffMatrix[row, row] += yCondCoeff
            coeffMatrix[row, row + below] += -2* yCondCoeff
            coeffMatrix[row, row + 2 * below] += yCondCoeff
        else:
            #Centered difference method
            coeffMatrix[row, row + below] += yCondCoeff
            coeffMatrix[row, row] += -2 * yCondCoeff
            coeffMatrix[row, row + above] += yCondCoeff
    
    #Convection
    for row in range(matrix2dSize):
        #remember that the air will be moved, before this is added
        botAirIndex = row
        plateIndex = row + matrix2dSize
        topAirIndex = row + 2 * matrix2dSize
        
        above = plateDisc["x"]
        convecting = convectingArrayBool(xDiscretizations, yDiscretizations)

        
        if convecting[row]:
            #plate to bottom air interaction, and moves along the x-axis
            coeffMatrix[botAirIndex + 1, botAirIndex] -= botConvCoeff
            coeffMatrix[botAirIndex + 1, plateIndex] += botConvCoeff

            #plate bottom air interaction, and the plate does not move
            coeffMatrix[plateIndex, plateIndex] -= botConvCoeff
            coeffMatrix[plateIndex, botAirIndex] += botConvCoeff
            
            #plate top air interaction
            coeffMatrix[plateIndex, plateIndex] -= topConvCoeff
            coeffMatrix[plateIndex, topAirIndex] += topConvCoeff
            
            #top air plate interaction, and moves along the y axis
            coeffMatrix[topAirIndex + above, topAirIndex] -= topConvCoeff
            coeffMatrix[topAirIndex + above, plateIndex] += topConvCoeff
            
        
    return csr_matrix(coeffMatrix)

def constructInverseM_Matrix(num2dDataPoints: int, 
                            botValue: float, 
                            midValue: float,
                            topValue: float) -> csr_matrix:
    return diags(np.concatenate([np.full(num2dDataPoints, botValue),
                                 np.full(num2dDataPoints, midValue),
                                 np.full(num2dDataPoints, topValue)]),
                 format="csr")

def constructAirMoveMatrix(matrix2dSize: int, plateDisc: dict) -> csr_matrix:
    
    matrix3dSize = matrix2dSize * 3
    transMatrix = lil_matrix((matrix3dSize, matrix3dSize))
    
    for botAirIndex in range(matrix2dSize):
        plateIndex = botAirIndex + matrix2dSize
        topAirIndex = botAirIndex + 2 * matrix2dSize
        
        #Plate does not move, and should therefore not be changed
        transMatrix[plateIndex, plateIndex] = 1
        
        if botAirIndex % plateDisc["x"] == 0:
            #First Column
            transMatrix[botAirIndex, botAirIndex] = 1
        else:
            transMatrix[botAirIndex, botAirIndex - 1] = 1
        
        if botAirIndex < plateDisc["x"]:
            #First row
            transMatrix[topAirIndex, topAirIndex] = 1
        else:
            transMatrix[topAirIndex, topAirIndex - plateDisc["x"]] = 1
    
    return csr_matrix(transMatrix)
    
def createPlateHeatMapVideo(heatMatrix, output_filename="plate_heatmap", 
                            frame_skip=1, 
                            vmin=0, vmax=50, 
                            save_gif=True, save_mp4=True):
    """
    Creates a heatmap animation of the plate's heat distribution over time and saves it as a GIF and/or MP4.

    Args:
        heatMatrix (np.ndarray): The heat matrix with shape (time, 3, x, y).
        output_filename (str): The base name of the output file (without extension).
        frame_skip (int): Number of frames to skip between each frame in the animation.
        vmin (float): Minimum temperature for the color scale.
        vmax (float): Maximum temperature for the color scale.
        save_gif (bool): Whether to save the animation as a GIF.
        save_mp4 (bool): Whether to save the animation as an MP4.
    """
    global dt

    # Extract the plate's heat distribution over time
    plate_heat = heatMatrix[:, 1, :, :]  # Shape: (time, x, y)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    cax = ax.imshow(plate_heat[0], cmap="jet", interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
    fig.colorbar(cax, ax=ax, label="Temperature (°C)")
    ax.set_title("Plate Heat Distribution Over Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Update function for animation
    def update(frame):
        cax.set_array(plate_heat[frame])
        ax.set_title(f"Plate Heat Distribution at t = {frame * dt:.2f} seconds")
        return cax,

    # Create the animation with skipped frames
    frames = range(0, plate_heat.shape[0], frame_skip)
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    # Save the animation as a GIF
    if save_gif:
        anim.save(f"{output_filename}.gif", writer=PillowWriter(fps=20))
        print(f"GIF saved as {output_filename}.gif")

    # Save the animation as an MP4
    if save_mp4:
        anim.save(f"{output_filename}.mp4", writer=FFMpegWriter(fps=20))
        print(f"MP4 saved as {output_filename}.mp4")

    plt.close(fig)

def createExhaustOverTimePlot(heatMatrix: np.ndarray, 
                              Discs: int, 
                              outputFileName: str):
    # Extract the exhaust temperatures (last column and row of bottom and top air layers)
    bot_exhaust = heatMatrix[Discs-1:, 0, 1:-1, -1]  # Bottom air layer, last column
    top_exhaust = heatMatrix[Discs-1:, 2, -1, 1:-1]  # Top air layer, last row

    # Calculate the average exhaust temperature over time
    average_bot_exhaust_temp = bot_exhaust.mean(axis=1)  # Average over the last column
    average_top_exhaust_temp = top_exhaust.mean(axis=1)  # Average over the last row

    # Plot the exhaust temperatures over time
    time_steps = np.arange(len(average_bot_exhaust_temp)) * dt  # Time in seconds
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, average_bot_exhaust_temp, label="Cold Exhaust Temperature", color="blue")
    plt.plot(time_steps, average_top_exhaust_temp, label="Hot Exhaust Temperature", color="red")

    
    plt.title(outputFileName)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.savefig(outputFileName)
    plt.clf()

def main(simTime: float, 
         plateProperties: dict, 
         botAirProperties: dict, 
         topAirProperties: dict, 
         Discretizations: dict = {"x": 20, "y": 20}) -> np.ndarray:

    global dt

    num2dDataPoints = Discretizations["x"] * Discretizations["y"]
    num3dDataPoints = num2dDataPoints * 3

    #plateProperties = plateProperties.copy()
    
    # Distribute discretizations for the x-direction
    xDiscretizations = distribute_discretizations(
        Discretizations["x"],
        plateProperties["x"],
        plateProperties["MountSize_x"]
    )

    # Distribute discretizations for the y-direction
    yDiscretizations = distribute_discretizations(
        Discretizations["y"],
        plateProperties["y"],
        plateProperties["MountSize_y"]
    )
    
    plateDiffs = {
        "dx": plateProperties["x"] / xDiscretizations["plate"],
        "dy": plateProperties["y"] / yDiscretizations["plate"],
        "dz": plateProperties["thickness"]
    }
    plateDiffs["area"] = plateDiffs["dx"] * plateDiffs["dy"]
    plateDiffs["volume"] = plateDiffs["area"] * plateDiffs["dz"]
    
    dt = min(Calc_dt(botAirProperties["botAirSpeed"],topAirProperties["topAirSpeed"],plateDiffs["dx"],plateDiffs["dy"]))
    
    timeIntervals = math.ceil(simTime / dt)
    
    botHeatVector = np.full(num2dDataPoints, plateProperties["startTemp"])
    for i in range(0,len(botHeatVector),Discretizations["x"]):
        botHeatVector[i] = botAirProperties["startTemp"]
    
    plateHeatVector = np.full(num2dDataPoints, plateProperties["startTemp"])
    
    topHeatVector = np.full(num2dDataPoints, plateProperties["startTemp"])
    for i in range(Discretizations["x"]):
        topHeatVector[i] = topAirProperties["startTemp"]
    
    heatVector = np.zeros((timeIntervals,num3dDataPoints))
    heatVector[0] = np.concatenate((botHeatVector,
                                    plateHeatVector,
                                    topHeatVector))
    # heatVector[0,num2dDataPoints + num2dDataPoints // 2 + 5] = 100 #Troubleshoot conduction
    
    
    
    botAirMValue = botAirProperties["density"] * botAirProperties["specific_heat_capacity"] * plateDiffs["area"] * botAirProperties["botAirHeight"]
    plateMValue = plateProperties["density"] * plateProperties["specific_heat_capacity"] * plateDiffs["volume"]
    topAirMValue = topAirProperties["density"] * topAirProperties["specific_heat_capacity"] * plateDiffs["area"] * topAirProperties["topAirHeight"]
    
    inverseM = constructInverseM_Matrix(num2dDataPoints,
                                       1/botAirMValue,
                                       1/plateMValue,
                                       1/topAirMValue)
    


    xCondCoeff = (plateProperties["thermal_conductivity"] * plateDiffs["volume"]) / (plateDiffs["dx"] ** 2)
    yCondCoeff = (plateProperties["thermal_conductivity"] * plateDiffs["volume"]) / (plateDiffs["dy"] ** 2)
    
    
    kMatrix = constructKMatrix(xCondCoeff,yCondCoeff,
                               botAirProperties["convection_coefficient"] * plateDiffs["area"], 
                               topAirProperties["convection_coefficient"] * plateDiffs["area"],
                               num2dDataPoints, Discretizations,
                               xDiscretizations, yDiscretizations)
    
    
    moveAirMatrix = constructAirMoveMatrix(num2dDataPoints, Discretizations)
    
    
    
    for t in range(timeIntervals-1):
        heatVector[t+1] = moveAirMatrix @ heatVector[t] + dt * (inverseM @ (kMatrix @ heatVector[t]))
        if  t % 10000 == 0:
            print(heatVector[t+1])  
    return heatVector





if __name__ == "__main__":
    
    # MATERIAL PROPERTIES - aluminum
    alu_properties = {
        "thermal_conductivity": 205.0,  # W/(m·K)
        "density": 2700.0,              # kg/m³
        "specific_heat_capacity": 900.0, # J/(kg·K)
        "thickness": 0.001,              # m
        "startTemp": 23,
        # dimensions
        "x": 0.08,
        "y": 0.08,

        "MountSize_x": 0.02, 
        "MountSize_y": 0.02,
    }

    #Material properties - cardboard
    cardboard_properties = {
        "thermal_conductivity": 0.05,  # W/(m·K)
        "density": 689.0,              # kg/m³
        "specific_heat_capacity": 1336.0, # J/(kg·K)
        "thickness": 0.00024,           # m
        "startTemp": 23,
        # dimensions
        "x": 0.08,
        "y": 0.08,

        "MountSize_x": 0.02, 
        "MountSize_y": 0.02,
    }
    
    # MATERIAL PROPERTIES - top atmospheric air
    top_alu_atmAir_properties = {
        "thermal_conductivity": 0.0257,  # W/(m·K)
        "density": 1.225,               # kg/m³ (at 15°C and 1 atm)
        "specific_heat_capacity": 1005.0, # J/(kg·K)
        "convection_coefficient": 21.92,    # h
        "startTemp": 40,

        "topAirSpeed": 2.0,  # m/s
        "topAirHeight": 0.005
    }
    
    # MATERIAL PROPERTIES - bot atmospheric air
    bot_alu_atmAir_properties = {
        "thermal_conductivity": 0.0257,  # W/(m·K)
        "density": 1.225,               # kg/m³ (at 15°C and 1 atm)
        "specific_heat_capacity": 1005.0, # J/(kg·K)
        "convection_coefficient": 20.43,    # h
        "startTemp": 15,

        "botAirSpeed": 2.0,  # m/s
        "botAirHeight": 0.005
    }
    
    # MATERIAL PROPERTIES - top atmospheric air
    top_karton_atmAir_properties = {
        "thermal_conductivity": 0.0257,  # W/(m·K)
        "density": 1.225,               # kg/m³ (at 15°C and 1 atm)
        "specific_heat_capacity": 1005.0, # J/(kg·K)
        "convection_coefficient": 21.92,    # h
        "startTemp": 40,

        "topAirSpeed": 2.0,  # m/s
        "topAirHeight": 0.005
    }
    
    # MATERIAL PROPERTIES - bot atmospheric air
    bot_karton_atmAir_properties = {
        "thermal_conductivity": 0.0257,  # W/(m·K)
        "density": 1.225,               # kg/m³ (at 15°C and 1 atm)
        "specific_heat_capacity": 1005.0, # J/(kg·K)
        "convection_coefficient": 20.43,    # h
        "startTemp": 15,

        "botAirSpeed": 2.0,  # m/s
        "botAirHeight": 0.005
    }
    
    """ #Running the simulation
    alu_heatVector = main(600.0,
                          alu_properties,
                          bot_alu_atmAir_properties,
                          top_alu_atmAir_properties)
    alu_heatMatrixShape = (len(alu_heatVector),3, 20, 20)
    alu_heatMatrix = alu_heatVector.reshape(alu_heatMatrixShape)

    print("Simulation done,\nProcessing started...")
    
    os.makedirs("alu_stuff", exist_ok=True)
    os.makedirs("karton_stuff", exist_ok=True)
    
    createExhaustOverTimePlot(heatMatrix=alu_heatMatrix, 
                              Discs=20, 
                              outputFileName=os.path.join("alu_stuff", "Exhaust Temperatures Over Time for aluminum"))
    
    
    # Create a heatmap animation and save as both GIF and MP4
    createPlateHeatMapVideo(
        alu_heatMatrix,
        output_filename=os.path.join("alu_stuff", "alu_plate_heatmap"),
        frame_skip= 400,  
        vmin=15,         # Minimum temperature for the scale
        vmax=40,         # Maximum temperature for the scale
        save_gif=True,   # Save as GIF
        save_mp4=True    # Save as MP4
    )
    
    createPlateHeatMapVideo(
        alu_heatMatrix,
        output_filename=os.path.join("alu_stuff", "alu_plate_heatmap_ZoomTemp"),
        frame_skip=400,  
        vmin=25,         # Minimum temperature for the scale
        vmax=30,         # Maximum temperature for the scale
        save_gif=False,   # Save as GIF
        save_mp4=True    # Save as MP4
    ) """

    #Running the simulation for karton
    karton_heatVector = main(3000.0,
                             cardboard_properties,
                             bot_karton_atmAir_properties,
                             top_karton_atmAir_properties)
    heatMatrixShape = (len(karton_heatVector),3, 20, 20)
    karton_heatMatrix = karton_heatVector.reshape(heatMatrixShape)

    print("Simulation done,\nProcessing started...")
    
    """ createExhaustOverTimePlot(heatMatrix=karton_heatMatrix, 
                              Discs=20,
                              outputFileName=os.path.join("karton_stuff", "Exhaust Temperatures Over Time for cardboard"))
    
    # Extract the exhaust temperatures (last column and row of bottom and top air layers)
    bot_exhaust = karton_heatMatrix[19:, 0, 1:-1, -1]  # Bottom air layer, last column
    top_exhaust = karton_heatMatrix[19:, 2, -1, 1:-1]  # Top air layer, last row

    # Calculate the average exhaust temperature over time
    average_bot_exhaust_temp = bot_exhaust.mean(axis=1)  # Average over the last column
    average_top_exhaust_temp = top_exhaust.mean(axis=1)  # Average over the last row

    # Time steps
    time_steps = np.arange(len(average_bot_exhaust_temp)) * dt  # Time in seconds

    # Plot cold exhaust temperature
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, average_bot_exhaust_temp, label="Cold Exhaust Temperature", color="blue")
    plt.title("Cold Exhaust Temperatures Over Time for Cardboard")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("karton_stuff", "Cold_Exhaust_Temperatures_Over_Time_for_Cardboard.png"))
    plt.clf()

    # Plot hot exhaust temperature
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, average_top_exhaust_temp, label="Hot Exhaust Temperature", color="red")
    plt.title("Hot Exhaust Temperatures Over Time for Cardboard")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("karton_stuff", "Hot_Exhaust_Temperatures_Over_Time_for_Cardboard.png"))
    plt.clf()
    
    # Create a heatmap animation and save as both GIF and MP4
    createPlateHeatMapVideo(
        karton_heatMatrix,
        output_filename=os.path.join("karton_stuff", "karton_plate_heatmap"),
        frame_skip=1000,  
        vmin=15,         # Minimum temperature for the scale
        vmax=30,         # Maximum temperature for the scale
        save_gif=True,   # Save as GIF
        save_mp4=True    # Save as MP4
    ) """
    
    createPlateHeatMapVideo(
        karton_heatMatrix,
        output_filename=os.path.join("karton_stuff", "karton_plate_heatmap_ZoomTemp"),
        frame_skip=1000,  
        vmin=25,         # Minimum temperature for the scale
        vmax=30,         # Maximum temperature for the scale
        save_gif=False,   # Save as GIF
        save_mp4=True    # Save as MP4
    )
