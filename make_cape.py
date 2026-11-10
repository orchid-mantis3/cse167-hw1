import json, math

N = 8 
ROUND = 0.35
BEND = 0.12

verts, faces = [], []
for j in range(N + 1):
    for i in range(N + 1):
        x = -1 + 2 * i / N
        y = -1 + 2 * j / N
        # blend betw. square and rounded corners
        rx = x * math.sqrt(1 - y * y / 2)
        ry = y * math.sqrt(1 - x * x / 2)
        px = (1 - ROUND) * x + ROUND * rx
        py = (1 - ROUND) * y + ROUND * ry
        # sides curve away from camera
        verts.append([px, py, -BEND * px * px])
for j in range(N):
    for i in range(N):
        a = j * (N + 1) + i
        b, c, d = a + 1, a + N + 2, a + N + 1
        faces += [[a, b, c], [a, c, d]]

json.dump({"vertices": verts, "faces": faces},
          open("scenes/hw3/cape.json", "w"))