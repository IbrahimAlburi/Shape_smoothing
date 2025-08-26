# Shape_smoothing

## Introduction
This code makes use of Chaikin's corner cutting algorithm but also attempts to ensure that important details are maintained. The overall goal is that the result is a polished cleaner version of the input structure.

## The Logic
Let's take as input a hand drawn image of a triangle: it's rough, noisy, contains a lot of imperfect strokes.

<img width="210" height="218" alt="triangle_border" src="https://github.com/user-attachments/assets/1014585b-98de-4145-a556-98505a7cf9f4" />

Given this structure, it's important to note that maybe some of these imperfect lines are intentional design. So it's essential that a smoothing tool is content-aware such that it wouldn't warp into something entirely different.

Take this triangle example that has passed through a certain number of iterations through Chaikin's algorithm:

<img width="273" height="220" alt="tri_Before_After2" src="https://github.com/user-attachments/assets/73fa9ae2-0d02-45a1-8956-785efe791f7b" />

The light grey represents the original contour while the black is post-corner-cutting. Notice how the corners are not as sharp as they originally were. Notice how the corners are cut, ultimately resulting in important features being lost. To avoid this I passed different algorithms before corner-cutting to ensure this doesn't happen.

This brings us to the first step of our 3-step process.
————————————————————————————————————————————————————
### Step 1
————————————————————————————————————————————————————

Step 1 consists of exaggerating features around the structure. This results in a more simpler looking structure with sharp corners. My implementation of this is made using Dijkstra's algorithm between every n-set of contour pixels. The Dijkstra algortihm is modified to prevent any diagonal paths from being returned. 
The algorithm t takes as input a start point and an n value indicating how many pixels we want to cross. Then it finds the shortest straight path from the image_contour[start] to image_contour[start + n]. The larger the n value the simpler the output looks. My example uses n = 7. This results in the following image:

<img width="192" height="216" alt="tri_Step_1" src="https://github.com/user-attachments/assets/c8ffad54-4f50-4cf1-a8ca-5dceaada6bcd" />

Notice how the shape maintains its strikingness, but is now simpler? This so that when applying corner-cutting, the structure is "molded" into it's original shape but with a smoother more clean output.

***Pointers***
  - Given how Dijkstra's algorithm isn't that efficient and that the paths are rather simple and predictable, I'm working on an re-implementation of this     algorithm possibly one that uses a modified bresenham line algorithm that returns straight paths.

————————————————————————————————————————————————————
### Step 2
————————————————————————————————————————————————————

After Step 1, it's important to understant that striking features might get cut during the corner-cutting process. This will result into shapes that lose their integrity. Decreasing the number of iterations to pass through Chaikin's algorithm would help, but it would also result in a less smooth shape. Increasing the number of iterations results in a smoother shape, but less striking features. To fix this I run the shape through the Ramer–Douglas–Peucker algorithm AKA RDP. This helps pinpoint sharp features that can later serve as pillars so that the important features don't fall out of place.

<img width="204" height="215" alt="tri_Step_2" src="https://github.com/user-attachments/assets/b3a53d39-5396-46d9-a753-943dcc01db5b" />

It's a little hard to see, but I highlight the RDP points in red. These points serve as pillars that will help maintain the integrity of the original shape after corner cutting.

————————————————————————————————————————————————————
### Step 3
————————————————————————————————————————————————————

The final step to the algorithm consists of applying corner cutting on the shape after all is said and done. Taking the pillars into account, the Chaikin algorithm is partitioned into sections between each pillar. So if we have Pillars A, B, and C then the algorithm would do Chaikin(all pixels between Pillar A -> B) then Chaikin(all pixels between Pillar B -> C) and finally Chaikin(all pixels between Pillar C -> A). This allows these striking features to avoid being cut by serving as constants. This results in the following output image:

<img width="194" height="208" alt="tri_Before_After" src="https://github.com/user-attachments/assets/b8e94715-64f6-420a-88b8-e18002d13d08" />

The grey is the original contour while black is the output image. Notice how the corners are intact as compared to the example provided above. 

Other examples are like this hand-drawn circle Image A is going directly through Chaikin corner cutting, and image B is going through the 3-step process:-

***Original Image:***

<img width="433" height="346" alt="circle_border" src="https://github.com/user-attachments/assets/54c99d5f-6031-48e3-a0b1-4cb0ffabe1de" />

**Image A:**

<img width="433" height="346" alt="Before_After2" src="https://github.com/user-attachments/assets/9286d6f3-1b2f-47ab-ac7e-020764832abc" />

**Image B:**

<img width="433" height="346" alt="Step_3" src="https://github.com/user-attachments/assets/b53ee72d-3abf-4f0a-a18c-fbe4cde1442a" />

The result is a smoother shape given the same number of iterations while also making sure that the shape doesn't lose much of it's striking features if any at all.




