import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def read_trackmate_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    tracks_data = []

    # Find the <AllSpots> element to gather all spots
    all_spots = {}
    spots_element = root.find(".//AllSpots")
    if spots_element is not None:
        for frame_element in spots_element.findall("SpotsInFrame"):
            for spot_element in frame_element.findall("Spot"):
                spot_id = spot_element.get("ID")
                all_spots[spot_id] = {
                    "frame": int(spot_element.get("FRAME")),
                    "x": float(spot_element.get("POSITION_X")),
                    "y": float(spot_element.get("POSITION_Y")),
                    "z": float(spot_element.get("POSITION_Z")),
                    "quality": float(spot_element.get("QUALITY"))
                }

    # Find the <AllTracks> element
    all_tracks = root.find(".//AllTracks")
    if all_tracks is not None:
        # Iterate through each <Track> element
        for track_element in all_tracks.findall("Track"):
            track_id = track_element.get("TRACK_ID")
            track_spots = []

            # Iterate through each <Edge> element within the track to find spot references
            for edge_element in track_element.findall("Edge"):
                spot_source_id = edge_element.get("SPOT_SOURCE_ID")
                spot_target_id = edge_element.get("SPOT_TARGET_ID")

                # Add the source and target spots to the track if they exist
                if spot_source_id in all_spots:
                    track_spots.append(all_spots[spot_source_id])
                if spot_target_id in all_spots:
                    track_spots.append(all_spots[spot_target_id])

            tracks_data.append({"track_id": track_id, "spots": track_spots})
    else:
        print("No <AllTracks> element found in the XML file.")

    return tracks_data

def plot_tracks(tracks):
    plt.figure(figsize=(10, 8))
    for track in tracks:
        x_coords = [spot['x'] for spot in track['spots']]
        y_coords = [spot['y'] for spot in track['spots']]
        plt.plot(x_coords, y_coords, marker='o', label=f"Track {track['track_id']}", markersize = 1)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Tracks Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

path = 'Timelapses/Regeneration_1.xml'
# Example usage:
tracks = read_trackmate_xml(path)

for track in tracks:
    print(f"Track ID: {track['track_id']}")
    for spot in track['spots']:
        print(f"  Spot ID: {spot.get('id', 'N/A')}, Frame: {spot['frame']}, X: {spot['x']}, Y: {spot['y']}, Z: {spot['z']}, Quality: {spot['quality']}")

plot_tracks(tracks)