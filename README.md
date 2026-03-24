Overview

This project implements an autonomous delivery drone system using:

MAVLink-based flight control
GPS waypoint navigation
Automated parcel drop using servo
Return-to-launch (RTL)

The system is designed to be modular, so it can later integrate:

ArUco / AprilTag precision landing
GPS-denied navigation


Core Features
1. Autonomous Takeoff
Arms drone
Takes off to defined altitude
Ensures safe initialization
2. GPS-Based Navigation
Moves to target coordinates using:
simple_goto()
Continuously monitors distance to target
3. Parcel Drop Mechanism
Uses MAVLink command:
MAV_CMD_DO_SET_SERVO
Controls servo motor to release payload
4. Return to Launch (RTL)
Automatically switches to RTL after delivery
System Architecture
Raspberry Pi (Companion Computer)
        ↓
Python Script (DroneKit + pymavlink)
        ↓
MAVLink Communication
        ↓
Cube Orange Flight Controller
        ↓
Drone (Motors + Servo)
Hardware Requirements
Flight Controller: Cube Orange / Pixhawk
Companion Computer: Raspberry Pi 5
GPS Module
Servo Motor (for payload drop)
Drone frame, ESC, motors, battery
Software Requirements
Python 3.x
DroneKit
pymavlink
NumPy
geopy


Installation
1. Clone Repository
git clone <your-repo-link>
cd delivery-drone
2. Install Dependencies
pip install dronekit pymavlink numpy geopy
3. Connect Flight Controller

Check connection:

ls /dev/ttyACM*

Run:

python3 script.py --connect /dev/ttyACM0
Mission Workflow
1. Connect to Drone
vehicle = connect(connection_string, baud=57600)
2. Takeoff
arm_and_takeoff(3)
3. Navigate to Target Location
goto_location(latitude, longitude)
4. Drop Parcel
drop_parcel()
5. Return Home
vehicle.mode = VehicleMode("RTL")


Key Functions
connectMyCopter()

Handles MAVLink connection via USB/serial.

arm_and_takeoff(altitude)

Performs pre-arm checks and safe takeoff.

goto_location(lat, lon)

Moves drone to specified GPS coordinates.

drop_parcel()

Triggers servo using MAVLink command.

my_mission()

Executes full mission pipeline:

Takeoff
Navigate
Drop payload
RTL
Important Notes
System currently depends on GPS navigation
No obstacle avoidance implemented
No precision landing yet
No vision-based corrections
Known Limitations
Accuracy limited by GPS (~2–5 meters)
No fallback if GPS signal is weak
No landing pad detection
Fixed waypoint (not dynamic)
Future Enhancements (Your Next Step)

You should extend this system with:

1. Precision Landing (HIGH PRIORITY)
ArUco / AprilTag detection
MAVLink LANDING_TARGET integration
2. GPS-Denied Navigation
Visual Odometry / SLAM
Optical Flow integration
3. Safety Systems
Fail-safe for signal loss
Battery monitoring
Emergency landing
4. Companion Computer Pipeline
Multithreaded vision system
Real-time pose estimation
MAVLink feedback loop
Use Cases
Autonomous delivery systems
Smart logistics
Campus delivery robots
Disaster supply drops
Safety Guidelines
Always test in SITL before real drone
Use low altitude for initial tests
Ensure open environment (no obstacles)
Verify servo operation before flight
Conclusion

This project establishes a baseline autonomous drone delivery system using:

MAVLink communication
GPS waypoint navigation
Payload deployment

It is designed to scale into a full GPS-denied intelligent drone system with vision integration.
