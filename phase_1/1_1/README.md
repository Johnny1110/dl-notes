# Phase - 1 (Uint 1)

<br>

---

<br>

## Guide

Here's the theory behind where to begin and why:

* Start with 3Blue1Brown's "Essence of Linear Algebra" — specifically the first 3-4 videos on **vectors, linear combinations, and matrix transformations**. The key insight these videos give you is that matrices aren't just grids of numbers; **they're transformations of space.** When you later see a neural network layer doing `W @ x + b,` you'll understand it as **"rotating, stretching, and shifting the input space"** rather than just "multiplying numbers."

* Then open a Jupyter notebook and play with NumPy alongside the videos. For example, after watching the video on matrix multiplication, try creating matrices and multiplying them yourself. See what happens to vectors when you transform them. **Visualize it with matplotlib.**

* The reason to pair video + code from Day 1 is that deep learning is inherently about building intuition for how numbers transform through operations. The earlier you make that connection between "math on paper" and "code that runs," the faster everything else falls into place.

<br>
<br>

## 3Blue1Brown Videos

1. [Vectors | Chapter 1, Essence of linear algebra](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=2)
2. [Linear combinations, span, and basis vectors | Chapter 2, Essence of linear algebra](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=3)
3. [矩陣與線性變換](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=4)
4. [矩陣乘法與線性變換復合聯繫](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=5) 👈 



```

| 0  -1 |   | 1  |     | -2 |
|       | x |    |  =  |    |
| 1   0 |   | 2  |     | 1  |



| 1   1 |   | 1  |     | 3  |
|       | x |    |  =  |    | (1, 1) x (1, 2) = 3
| 0   1 |   | 2  |     | 2  | (0, 1) x (1, 2) = 2 

```

