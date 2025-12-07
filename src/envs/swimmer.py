import os
import argparse

import gymnasium as gym


def generate_swimmer_xml(n_links=5, link_len=0.1):
    header = """
<mujoco model="n_swimmer">
  <compiler angle="radian"/>
  <default>
    <joint armature="0.1" damping="1.0"/>
    <geom type="capsule" size="0.05 0.1" density="1000"/>
  </default>

  <worldbody>
    <body name="torso" pos="0 0 0">
      <geom size="0.05 0.1"/>
    """

    bodies = []
    for i in range(n_links):
        bodies.append(
            f"""
        <body name="link{i}" pos="{link_len} 0 0">
          <joint name="joint{i}" type="hinge" axis="0 0 1" range="-100 100"/>
          <geom/>
        """
        )

    # close body nesting
    xml = header
    for b in bodies:
        xml += b
    xml += "</body>" * n_links
    xml += (
        """
    </body>
  </worldbody>

  <actuator>
    """
        + "\n".join(
            [f'<motor joint="joint{i}" ctrlrange="-1 1"/>' for i in range(n_links)]
        )
        + """
  </actuator>
</mujoco>
"""
    )
    return xml


def parse_args():
    parser = argparse.ArgumentParser(description="Generate swimmer Mujoco XMLs.")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save swimmer XML files (default: current directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for n_links in [6, 15]:
        xml = generate_swimmer_xml(n_links)
        xml_filename = f"swimmer{n_links}.xml"
        xml_path = os.path.join(out_dir, xml_filename)
        with open(xml_path, "w") as f:
            f.write(xml)

        env = gym.make("Swimmer-v5", xml_file=xml_path)
        env.close()


if __name__ == "__main__":
    main()
