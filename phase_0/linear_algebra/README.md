# 線性代數

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

### How to run jupyter notebook to practice metrics and vectors:

```bash
jupyter notebook
```

<br>
<br>


## 3Blue1Brown Videos

1. [向量究竟是什么](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=2) -> [實戰](1_1_vectors/vectors.ipynb)
2. [线性组合.张成的空间与基](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=3) -> [實戰](1_2_linear_combination/linear_combination.ipynb)
3. [矩陣與線性變換](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=4) -> [實戰](1_3_linear_transformation/linear_transformation.ipynb)
4. [矩陣乘法與線性變換復合聯繫](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=5) -> [實戰](1_4_matrix_multiplication/matrix_multiplication.ipynb)

    矩陣乘法不符合交換律，但符合結合律:

    * AB != BA
    * (AB)C = A(BC)

    4-附錄. [三維空間線性變換](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=6) 
  
5. [行列式 `det()`](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=7) -> [實戰](1_5_determinant/determinant_playground.ipynb)
6. [逆矩阵、列空间、秩与零空间](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=8) -> [實戰](1_6_inverse_colspace_rank/inverse_colspace_rank.ipynb)
7. [点积与对偶性](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=10) -> [實戰](1_7_dot_product_duality/dot_product_duality.ipynb) 👈
8. [08-01 叉积的标准介绍](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=11) -> [實戰](1_8_cross_product/cross_product.ipynb)
    [08-02 以线性变换的眼光看叉积](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=12) -> [實戰](1_8_2_cross_product_linear_view/cross_product_linear_view.ipynb)

9. [基变换](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=13) -> [實戰](1_9_change_of_basis/change_of_basis.ipynb)
10. [特征向量与特征值](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=14) -> [實戰](1_10_eigenvectors_eigenvalues/eigenvectors_eigenvalues.ipynb)
11. [抽象向量空间](https://www.bilibili.com/video/BV1ib411t7YR?spm_id_from=333.788.videopod.episodes&vd_source=9780a181ac9f1fee5f680f255ee5bc73&p=15) -> [實戰](1_11_abstract_vector_spaces/abstract_vector_spaces.ipynb)


<br>

---

<br>


