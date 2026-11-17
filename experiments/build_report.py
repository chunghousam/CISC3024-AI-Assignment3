"""Build the CISC3024 AI Assignment 3 report from the saved experiment summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUMMARY_PATH = ROOT / "results" / "summary.json"
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "AI_Assignment3_Report.docx"
SOURCE_URL = "https://github.com/chunghousam/CISC3024-AI-Assignment3"


def pct(value: float, std: float | None = None) -> str:
    if std is None:
        return f"{100 * value:.1f}%"
    return f"{100 * value:.1f}% ± {100 * std:.1f}%"


def set_run_font(run, size=12, bold=False, italic=False, color=None):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_text(paragraph, text, size=12, bold=False, italic=False):
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return run


def paragraph(doc, text, size=12, bold=False, italic=False, align="left", space_after=8, first_line=True):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    if first_line and align == "justify":
        p.paragraph_format.first_line_indent = Cm(0.75)
    add_text(p, text, size=size, bold=bold, italic=italic)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    add_text(p, text, size=16 if level == 1 else 13, bold=True)
    return p


def caption(doc, text):
    return paragraph(doc, text, size=11, italic=True, align="left", space_after=10, first_line=False)


def add_picture(doc, name, width=6.3):
    path = FIGURES / name
    if not path.exists():
        raise FileNotFoundError(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))


def shade_header(cell):
    tc = cell._tePr if hasattr(cell, "_tePr") else cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {
        qn("w:fill"): "1F4E79",
        qn("w:val"): "clear",
    })
    tc_pr.append(shd)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(header)
        set_run_font(run, size=10, bold=True, color=RGBColor(255, 255, 255))
        shade_header(cell)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
            add_text(p, str(value), size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def configure(doc):
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_text(footer, "CISC3024 Pattern Recognition  ·  AI Assignment 3", size=9, italic=True)


def build():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    selected = summary["selected_summary"]
    curves = summary["test_curves_unweighted_big"]
    noise = summary["noise"]
    block = summary["block"]
    weight = summary["weighting_k8"]
    recon = summary["reconstruction"]
    params = summary["parameter_study"]["rows"]
    times = summary["fit_seconds"]

    def best_curve(name):
        curve = curves[name]
        index = max(range(len(curve["k"])), key=lambda i: curve["mean"][i])
        return curve["k"][index], curve["mean"][index], curve["std"][index]

    doc = Document()
    configure(doc)

    paragraph(doc, "University of Macau", size=14, bold=True, align="center", space_after=2, first_line=False)
    paragraph(doc, "CISC3024 Pattern Recognition", size=14, align="center", space_after=2, first_line=False)
    paragraph(doc, "AI Assignment 3", size=20, bold=True, align="center", space_after=6, first_line=False)
    paragraph(
        doc,
        "Bilateral Generalized 2DPCA for Face Recognition",
        size=16, bold=True, align="center", space_after=10, first_line=False,
    )
    for line in (
        "Name: CHUNG HOU SAM",
        "Student ID: UC325095",
        "Date: November 2026",
        "Tool: a Cursor coding agent wrote the code, ran the experiments, and drafted this report",
    ):
        paragraph(doc, line, align="center", space_after=2, first_line=False)
    paragraph(doc, "", space_after=6, first_line=False)

    paragraph(
        doc,
        "This report studies a 2023 principal-component method for images and compares it with three classical baselines on the Olivetti face dataset. The selected method is the real-matrix form of bilateral generalized two-dimensional quaternion PCA (BiG2DQPCA). On clean images, supervised LDA is the most accurate recognizer. The bilateral PCA model matches ordinary PCA and 2DPCA, stores a smaller feature than 2DPCA, and does not by itself solve large block occlusion.",
        align="justify",
    )

    # ------------------------------------------------------------------
    heading(doc, "1. How the AI found the algorithm")
    paragraph(
        doc,
        "The assignment asks for a recent PCA or LDA method with a computer-vision application, and it asks the agent to do the search, the programming, and the writing. The search started from papers published in 2023–2025 rather than from the original eigenfaces or Fisherfaces papers.",
        align="justify",
    )
    paragraph(
        doc,
        "Three recent LDA papers were serious candidates. Liu, Feng, Xiu, and Liu (Pattern Recognition, 2024) propose a robust sparse LDA for image classification. Zhou, Zhang, Zeng, Zhang, and Fang (Pattern Recognition, 2024) propose latent LDA with isometric structural learning. A follow-up by Liu and co-authors (IEEE Transactions on Emerging Topics in Computational Intelligence, 2025) studies an efficient robust sparse LDA. All three match the assignment: they are recent, they are LDA, and they target image classification. Their full optimization procedures are not in an open manuscript, so an implementation would have meant guessing missing update rules. That is not a reliable way to complete the experiment.",
        align="justify",
    )
    paragraph(
        doc,
        "The search then stayed with methods whose algorithms are public. F-norm 2DPCA (Wang, Gao, Gao, and Nie, AAAI 2017) is fully specified and is robust to outliers, but it is no longer recent. Yang’s 2DPCA (2004) is the right baseline, not the selected method. The paper that was kept is Zhao, Jia, Gong, and Zhang, “Data-Driven Bilateral Generalized Two-Dimensional Quaternion Principal Component Analysis with Application to Color Face Recognition” (arXiv:2306.07045, 2023; https://arxiv.org/abs/2306.07045). It is a PCA method, it is aimed at face recognition, and Algorithms 2–4 are written out: an iterative solver for Lp norms, a weighted feature matrix, and a reconstruction formula.",
        align="justify",
    )
    paragraph(
        doc,
        "The original model encodes a color pixel as a pure quaternion so that the red, green, and blue channels stay coupled. A grayscale image is the same object with the three imaginary parts equal to zero, and the algebra collapses to ordinary matrix arithmetic. This report implements that real special case and calls it BiG2DPCA. The color databases used in the paper (AR, Color FERET, Faces95, and Georgia Tech) are large or access-restricted. Olivetti faces are public, small enough to rerun, and still a standard face-recognition benchmark. The numbers below are therefore a new experiment on grayscale faces, not a reproduction of the paper’s color-face table.",
        align="justify",
    )

    # ------------------------------------------------------------------
    heading(doc, "2. Algorithm description")
    heading(doc, "2.1 Eigenfaces (PCA)", level=2)
    paragraph(
        doc,
        "Turk and Pentland represent each face by a vector. After subtracting the mean face, PCA keeps the directions of largest variance. If the centered training vectors are the rows of X, those directions are the leading right singular vectors of X. A test face is centered with the same mean and projected onto the kept vectors. The feature is a short coefficient vector. Reconstruction adds the mean back to the partial expansion. PCA is optimal for squared reconstruction error in the vectorized pixel space, but the covariance it would form explicitly is 4096 by 4096 for a 64×64 image, so the implementation uses the thin SVD of the data matrix instead.",
        align="justify",
    )
    heading(doc, "2.2 Two-dimensional PCA", level=2)
    paragraph(
        doc,
        "Yang, Zhang, Frangi, and Yang (2004) do not vectorize the image. For centered images A_i of size m×n, 2DPCA builds the n×n image covariance G = Σ_i A_i^T A_i and takes the k eigenvectors V that belong to the largest eigenvalues. The feature of an image is the matrix Y = A V, which has shape m×k. Distance between two faces is the Euclidean distance between these matrices, which is the same as flattening them. Reconstruction is A V V^T plus the mean. The covariance is only 64×64 here, so the eigendecomposition is cheap. The price is that the feature still has one full image dimension: k = 12 already means 64×12 = 768 numbers.",
        align="justify",
    )
    heading(doc, "2.3 Linear discriminant analysis", level=2)
    paragraph(
        doc,
        "LDA uses the identity labels. It looks for directions that keep images of the same person close and push class means apart. With 64×64 images and 200 training faces, the within-class scatter is singular: its rank is at most the number of samples minus the number of classes. The implementation follows the stable SVD form of LDA. Class-centered training images are scaled by their within-class standard deviation, factored by an SVD, and the class means are then projected into that whitened space. A second SVD of the scaled class means produces at most C−1 discriminant axes. A face is centered by the global mean and multiplied by those axes.",
        align="justify",
    )
    paragraph(
        doc,
        "The classical Fisherfaces pipeline, PCA down to N−C dimensions followed by an explicit within-class inverse, was implemented first and checked against scikit-learn. On the first split it reached only about 53% test accuracy, and scikit-learn’s PCA-then-LDA pipeline was similarly weak. scikit-learn’s SVD solver on the same split reached 97.5%. The reported LDA baseline is that SVD solver, reimplemented in NumPy and checked until the projected coordinates matched scikit-learn with cosine 1.0. The difference matters: discarding all but N−C principal components throws away discriminant directions that the SVD solver keeps.",
        align="justify",
    )
    heading(doc, "2.4 Bilateral generalized 2DPCA", level=2)
    paragraph(
        doc,
        "BiG2DPCA projects an image from both sides. For centered images and two sets of vectors, the training criterion in the 2023 paper is to maximize",
        align="justify",
    )
    paragraph(
        doc,
        "Σ_i ( Σ_a ||u_a^T A_i||_s^s  +  Σ_b ||A_i v_b||_s^s )",
        align="center", first_line=False, italic=True,
    )
    paragraph(
        doc,
        "with ||u_a||_p = 1, ||v_b||_p = 1, and orthogonality between different vectors on the same side. The exponents satisfy s ≥ 1 and p > 0. Squared Frobenius energy is the special case s = p = 2. A smaller p down-weights the influence of a single large coordinate and is the paper’s route to robustness.",
        align="justify",
    )
    paragraph(
        doc,
        "The two sides separate, so the right vectors can be computed without the left vectors. Algorithm 2 builds them one at a time. A random vector with unit Lp norm is updated until the objective stops changing. Each update multiplies every image by the current vector, replaces that product by |z|^{s−1} sign(z), and multiplies back by the transposed image. The resulting vector is then normalized in a way that depends on p: the dual exponent q = p/(p−1) when p > 1, a signed one-hot vector when p = 1, the sign pattern when p is infinite, and an extra weight from the initial vector when p < 1. The new vector is orthonormalized against the previous ones by QR, its objective value is stored as a weight, and the images are deflated by F ← F(I − vv^T). The left vectors are the same procedure applied to A_i^T.",
        align="justify",
    )
    paragraph(
        doc,
        "When s = p = 2 the iteration is power iteration on Σ_i A_i^T A_i. Its solution is exactly the leading eigenvectors of that matrix, and the stored weights are the eigenvalues. The same statement on the transposed images gives the left basis. The experiments use this closed form for s = p = 2, because the paper’s relative tolerance of 10^{-4} can stop the power iteration a few degrees short of the eigenvector when two eigenvalues are close. A separate run of Algorithm 2 with tolerance 10^{-8} is kept as a check and as the convergence plot. For every other (s, p) pair the code runs Algorithm 2 itself.",
        align="justify",
    )
    paragraph(
        doc,
        "The feature of a centered image A is the k×k matrix P = U^T A V. Four weightings are compared. Unweighted projection uses U and V as they are. Left weighting replaces U by U diag(w_left), right weighting scales V, and bilateral weighting scales both. The paper’s data-driven step chooses the weighting on held-out images rather than fixing it in advance. Classification flattens P and uses 1-nearest neighbor. Reconstruction does not need the weights: they cancel, and the restored image is U U^T A V V^T plus the mean.",
        align="justify",
    )
    heading(doc, "2.5 Why nearest neighbor", level=2)
    paragraph(
        doc,
        "Both Yang’s 2DPCA paper and the BiG2DQPCA paper classify with nearest neighbor under Euclidean distance. The same rule is used for every method, including PCA and LDA, so the comparison is about the feature rather than about a second tuned classifier. With 40 identities the chance level is 2.5%.",
        align="justify",
    )

    # ------------------------------------------------------------------
    heading(doc, "3. How the AI implemented the algorithm")
    paragraph(
        doc,
        "All of the code is in this project. src/big2dpca.py contains Algorithm 2 and the closed-form s = p = 2 path. src/baselines.py contains eigenfaces, 2DPCA, and SVD LDA. src/data_utils.py loads Olivetti faces, splits each subject separately, adds noise, and evaluates 1-nearest neighbor. experiments/run_experiments.py runs the protocol below and writes results/summary.json together with the figures. experiments/build_report.py reads that JSON and produces this document, so the tables are filled from the run rather than typed in.",
        align="justify",
    )
    paragraph(
        doc,
        "The first implementation check is algebraic. On random 12×9 matrices, the closed-form bases match the covariance eigenvectors with principal angle 0. A tight run of Algorithm 2 matches the same right subspace to 0.058 degrees. That check is executed at the start of every experiment run and aborts if the angle exceeds half a degree. The second check is the LDA comparison with scikit-learn described in Section 2.3. The third check is visual: eigenfaces should look like faces, a 2DPCA reconstruction of a block should smear horizontally, and a bilateral reconstruction should smear in both directions. Those patterns appear in the figures below.",
        align="justify",
    )
    add_picture(doc, "convergence.png", width=5.6)
    caption(
        doc,
        "Figure 1. Objective of Algorithm 2 on the first Olivetti training split, s = p = 2. Both the left and the right vector level off within about ten iterations. The closed form used in the timed experiments is the limit of these curves.",
    )

    # ------------------------------------------------------------------
    heading(doc, "4. Experiment settings and results")
    heading(doc, "4.1 Data and protocol", level=2)
    paragraph(
        doc,
        "Olivetti faces, as distributed by scikit-learn, contain 400 grayscale images of 40 people, 10 images each, already scaled to 64×64 with pixel values in [0, 1]. Figure 2 shows the first five stored images of eight subjects. Expression, glasses, and a small amount of pose change are visible; the background and alignment are controlled.",
        align="justify",
    )
    add_picture(doc, "sample_faces.png", width=5.8)
    caption(doc, "Figure 2. Eight Olivetti subjects, five images each.")
    paragraph(
        doc,
        "Each of five splits uses an independent shuffle inside every subject, with seeds 2026 through 2030. Five images train, two validate, and three test, so each split has 200 training images, 80 validation images, and 120 test images. Every identity appears in every part. The subspace size is chosen by validation accuracy. Ties keep the smaller size. For BiG2DPCA the validation set also chooses one of the four weightings. Test images are never used to choose a setting. The curves in Figure 3 are test accuracy against size, averaged over the five splits, and are descriptive: the size at the top of a curve was not hidden from the test set. Table 1 is the stricter, validation-chosen number.",
        align="justify",
    )
    paragraph(
        doc,
        "Corruption is applied only to test images, after the model has been fit on clean training faces. Gaussian noise has standard deviation 0.20 and is clipped to [0, 1]. Salt-and-pepper noise replaces 25% of the pixels by 0 or 1. A block occlusion is a black square of side 8, 16, 24, or 32 pixels at a random position. The recognizer is not told where the block is.",
        align="justify",
    )

    heading(doc, "4.2 Clean recognition", level=2)
    paragraph(
        doc,
        f"Figure 3 shows that every method is far above the 2.5% chance level once it has enough components, and that extra components stop helping. PCA rises from {pct(curves['PCA']['mean'][0])} at 5 eigenfaces to {pct(*best_curve('PCA')[1:])} at {best_curve('PCA')[0]} eigenfaces. 2DPCA is already {pct(curves['2DPCA']['mean'][0])} with a single projection vector, dips at two vectors, and reaches {pct(*best_curve('2DPCA')[1:])} at {best_curve('2DPCA')[0]} vectors. Unweighted BiG2DPCA needs several vectors on each side: one vector per side is only {pct(curves['BiG2DPCA']['mean'][0])}, while k = {best_curve('BiG2DPCA')[0]} reaches {pct(*best_curve('BiG2DPCA')[1:])}. LDA is poor in one or two dimensions, because 40 classes cannot be separated there, and then climbs to {pct(*best_curve('LDA')[1:])} at {best_curve('LDA')[0]} directions.",
        align="justify",
    )
    add_picture(doc, "accuracy_vs_components.png", width=6.3)
    caption(doc, "Figure 3. Mean test accuracy and sample standard deviation over five splits. BiG2DPCA in this figure is unweighted.")

    paragraph(
        doc,
        "Choosing the size on the validation split gives the test accuracies in Table 1. LDA remains well ahead. PCA, 2DPCA, and BiG2DPCA are within about one point of each other. The validation set has only two images per person, so the chosen size moves around: PCA selects 20, 60, 60, 30, and 20 components, and BiG2DPCA selects either no weighting or right weighting. The test accuracy is nevertheless stable for LDA (sample standard deviation under one point) and within about three points for the unsupervised methods.",
        align="justify",
    )
    choice_text = {
        "PCA": ", ".join(str(c) for c in selected["PCA"]["choices"]),
        "2DPCA": ", ".join(str(c) for c in selected["2DPCA"]["choices"]),
        "LDA": ", ".join(str(c) for c in selected["LDA"]["choices"]),
        "BiG2DPCA": "; ".join(
            f"{pair[0]}/{pair[1]}" for pair in selected["BiG2DPCA"]["choices"]
        ),
    }
    add_table(
        doc,
        ["Method", "Validation-chosen test accuracy", "Chosen settings"],
        [
            [name, pct(selected[name]["mean"], selected[name]["std"]), choice_text[name]]
            for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")
        ],
    )
    caption(doc, "Table 1. Test accuracy when the model size, and for BiG2DPCA the weighting, is chosen on validation data. Mean ± sample standard deviation over five splits. BiG2DPCA settings are listed as k/weighting.")
    paragraph(
        doc,
        "The feature sizes explain why a similar accuracy can still favor one representation. At k = 16, BiG2DPCA stores a 16×16 matrix, 256 numbers. 2DPCA at 12 vectors stores 64×12 = 768 numbers for a similar clean accuracy. PCA at its best curve point stores 120 coefficients and is about one point lower. LDA uses at most 39 coefficients and is about nine points higher, but it is the only method that sees the training labels. On this clean, aligned dataset the label is worth more than a change from PCA to bilateral 2DPCA.",
        align="justify",
    )
    add_picture(doc, "eigenfaces.png", width=5.2)
    caption(doc, "Figure 4. The first eight eigenfaces from split 1. They are global face patterns, with lighting and glasses appearing in the leading components.")
    add_picture(doc, "bilateral_bases.png", width=5.2)
    caption(doc, "Figure 5. The first eight BiG2DPCA bases on the same split, drawn as the rank-1 images u_i v_i^T. Each basis is separable across rows and columns, so a single basis cannot copy a full eigenface. Later bases pick up high-frequency grid structure.")
    add_picture(doc, "lda_scatter.png", width=5.4)
    caption(doc, "Figure 6. Eight training identities in the first two LDA coordinates of split 1. Some identities form tight groups, and some overlap. The accuracy curve shows that the later discriminant coordinates are what separate all 40 people.")

    heading(doc, "4.3 Data-driven weighting", level=2)
    paragraph(
        doc,
        f"Figure 7 fixes k = 8 and compares the four weightings on all five splits. Unweighted features are best on both validation ({pct(weight['none']['val']['mean'], weight['none']['val']['std'])}) and test ({pct(weight['none']['test']['mean'], weight['none']['test']['std'])}). Right weighting is close ({pct(weight['right']['test']['mean'], weight['right']['test']['std'])} test). Left weighting falls to {pct(weight['left']['test']['mean'], weight['left']['test']['std'])}, and weighting both sides falls to {pct(weight['both']['test']['mean'], weight['both']['test']['std'])}. The raw objective values differ by a large factor between the first vector and the later ones. Using them as multipliers on both sides lets the first coefficient dominate the Euclidean distance. The validation set notices this: in Table 1 it never selects left or bilateral weighting. It does sometimes select right weighting at a different k, where the validation accuracy was higher than the unweighted alternative. The procedure in the paper is doing the right thing. The useful conclusion on Olivetti is that the unweighted feature is already the strong one.",
        align="justify",
    )
    add_picture(doc, "weighting.png", width=5.6)
    caption(doc, "Figure 7. BiG2DPCA weighting at k = 8. Bars are means over five splits, with sample standard deviations.")

    heading(doc, "4.4 Noise and occlusion", level=2)
    paragraph(
        doc,
        "Table 2 uses the validation-chosen model from Table 1 and corrupts only the test image. Mild Gaussian noise barely moves PCA, 2DPCA, or BiG2DPCA, and it costs LDA about four points. Salt-and-pepper noise costs every method roughly 15 to 20 points. A 24×24 black block, which covers 14% of the image and often covers an eye or the mouth, collapses recognition for all four methods. LDA is no longer clearly best, and BiG2DPCA is the weakest of the four under that block.",
        align="justify",
    )
    noise_rows = []
    labels = [
        ("clean", "Clean"),
        ("gaussian_0.20", "Gaussian, σ = 0.20"),
        ("salt_pepper_0.25", "Salt-and-pepper, 25%"),
        ("block_24", "Block, 24×24"),
    ]
    for key, label in labels:
        noise_rows.append([label] + [pct(noise[key][name]["mean"], noise[key][name]["std"]) for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")])
    add_table(doc, ["Test condition", "PCA", "2DPCA", "LDA", "BiG2DPCA"], noise_rows)
    caption(doc, "Table 2. Accuracy under test-image corruption at the validation-chosen operating point. Mean ± sample standard deviation over five splits.")
    add_picture(doc, "noise_bars.png", width=6.3)
    caption(doc, "Figure 8. The same comparison as Table 2.")
    paragraph(
        doc,
        "Figure 9 varies only the block size, with a fresh random position for each size. Accuracy stays high for an 8×8 block. At 16×16, PCA still has a clear lead. At 24×24 and 32×32 every curve is near or below 30%, and a 32×32 block covers a quarter of the face. The Lp extension was supposed to be the robust part of the 2023 model. On one split, with k fixed at 8, s = 1 or p = 1 does raise block accuracy relative to s = p = 2, but the gain is from 25% to at most 37.5%, while clean accuracy falls from 87.5% to as low as 64.2% (Table 3). That is one split, so it is a sensitivity check rather than a second benchmark. It is enough to see that changing the norm does not remove a large missing region. None of these methods has an occlusion mask. A black square is treated as if it were part of the face.",
        align="justify",
    )
    add_picture(doc, "block_sweep.png", width=5.8)
    caption(doc, "Figure 9. Test accuracy against the side length of a black square. The model size is still the one chosen on clean validation images.")
    param_rows = []
    for row in params:
        param_rows.append([
            f"s = {row['s']:g}, p = {row['p']}",
            "failed" if row["clean_accuracy"] is None else pct(row["clean_accuracy"]),
            "failed" if row["block_accuracy"] is None else pct(row["block_accuracy"]),
        ])
    add_table(doc, ["Norms", "Clean test", "24×24 block"], param_rows)
    caption(doc, "Table 3. BiG2DPCA on the first split only, k = 8, unweighted. The s = p = 2 row is the closed form. The other rows are Algorithm 2.")
    add_picture(doc, "parameter_study.png", width=6.0)
    caption(doc, "Figure 10. The same one-split sensitivity study as Table 3.")

    heading(doc, "4.5 Reconstruction", level=2)
    paragraph(
        doc,
        "Figure 11 plots the mean reconstruction ratio 1 − ||A − Â||_F / ||A||_F on the training images of split 1, against the number of stored coefficients rather than against k. Those two counts are different things: a 2DPCA model with k vectors stores 64k coefficients, and a BiG2DPCA model stores k². PCA, which is optimal for this vectorized squared error, is the most economical. At 120 coefficients its ratio is "
        f"{recon['PCA'][recon['PCA_grid'].index(120)]:.3f}. BiG2DPCA at k = 16 stores 256 coefficients and reaches {recon['BiG2DPCA'][-1]:.3f}. 2DPCA can post a high ratio, but only after storing a long feature; at k = 16 it stores 1024 coefficients for a ratio of {recon['2DPCA'][-1]:.3f}.",
        align="justify",
    )
    add_picture(doc, "reconstruction_curve.png", width=5.8)
    caption(doc, "Figure 11. Reconstruction ratio on the training images of split 1. The horizontal axis is the number of coefficients kept in the feature, not the number of projection vectors.")
    paragraph(
        doc,
        "Figure 12 applies the clean subspace to block-occluded test faces. PCA fills the hole with a blurry face-like pattern and often changes the apparent identity, because the black pixels pull the coefficients. 2DPCA spreads the hole horizontally, since each image row is reconstructed from the same right vectors. BiG2DPCA spreads it in both directions and leaves a cross or a dark blob. These pictures match the formulas, and they explain the recognition drop: the subspace is completing the missing pixels from the training variation, not detecting and ignoring them.",
        align="justify",
    )
    add_picture(doc, "reconstruction_gallery.png", width=6.2)
    caption(doc, "Figure 12. Six test faces from split 1. From top to bottom: the original, the same image with a 24×24 block, PCA reconstruction (40 eigenfaces), 2DPCA reconstruction (8 vectors), and BiG2DPCA reconstruction (k = 8).")

    heading(doc, "4.6 Runtime", level=2)
    paragraph(
        doc,
        "Table 4 is wall-clock time to fit one training split of 200 images on the machine used for this assignment. 2DPCA and the closed-form BiG2DPCA solve a 64×64 eigenproblem and finish in a few hundredths of a second. PCA and LDA factor a 200×4096 matrix and take longer, still well under a second. One run of Algorithm 2 for a non-Euclidean (s, p) pair and k = 8 took between 0.48 and 0.79 seconds. The bilateral model is not expensive on this image size. The experiment’s cost is the repeated nearest-neighbor evaluation, not the decomposition.",
        align="justify",
    )
    add_table(
        doc,
        ["Method", "Fit time (seconds)"],
        [[name, f"{times[name]['mean']:.3f} ± {times[name]['std']:.3f}"] for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")],
    )
    caption(doc, "Table 4. Mean fit time over the five splits. BiG2DPCA is the s = p = 2 closed form.")

    # ------------------------------------------------------------------
    heading(doc, "5. What was learned from this AI assignment")
    paragraph(
        doc,
        "The useful part of having an agent do the assignment was the checking, not the first draft. Three checks changed the result.",
        align="justify",
    )
    paragraph(
        doc,
        "The first was the special case s = p = 2. Algorithm 2 is then power iteration, so its vectors have to be the covariance eigenvectors. Stopped at the paper’s tolerance of 10^{-4}, the subspace was still several degrees off on a small random problem, because the objective can look flat before the vector has finished rotating. The fix was to use the eigen-decomposition for this special case and to keep a tighter iterative run as a witness. Without that comparison, a plausible implementation would have been quietly wrong.",
        align="justify",
    )
    paragraph(
        doc,
        "The second was the LDA baseline. A textbook Fisherfaces pipeline, PCA to N−C and then an inverse of the within-class scatter, looked correct and agreed with another library’s PCA-then-LDA path, but it recognized only about half of the first test split. The SVD form of LDA recognized 97.5% of that same split. For images, “LDA” is not one numerical procedure. The small-sample-size problem is real, and how it is removed changes the accuracy by tens of points.",
        align="justify",
    )
    paragraph(
        doc,
        "The third was the gap between a paper’s claim and a new dataset. BiG2DQPCA is motivated by color, occlusion, and a learned weighting. On clean Olivetti faces the unweighted s = p = 2 model is simply a compact bilateral PCA: it ties PCA and 2DPCA, uses fewer coefficients than 2DPCA, and loses to LDA because LDA sees the labels. Under a large black block it is the least accurate of the four, and the Lp settings recover only a little of that loss while damaging the clean result. The reconstruction pictures show why. A linear subspace fills a hole; it does not ignore it. Recent does not mean better on every protocol, and a color-quaternion paper does not automatically transfer to grayscale 1-nearest-neighbor recognition.",
        align="justify",
    )
    paragraph(
        doc,
        "Two practical limits should stay attached to the numbers. The validation set is small, so the chosen k is noisy even though the test average is stable. The occlusion is synthetic and unaligned with the face. A stronger conclusion about robustness would need the paper’s color databases, a mask or a robust loss that can reject pixels, and more than one split for the Lp comparison. Those are extensions, not what this run shows.",
        align="justify",
    )

    # ------------------------------------------------------------------
    heading(doc, "6. Webpage of the source code")
    paragraph(
        doc,
        "The agent wrote every source file. The project is published at",
        align="justify",
    )
    paragraph(doc, SOURCE_URL, align="center", bold=True, first_line=False, space_after=8)
    paragraph(
        doc,
        "The repository contains src/, experiments/, requirements.txt, README.md, the saved figures, results/summary.json, and this report. The Olivetti files are not included; scikit-learn downloads them on the first run. From the project directory the experiment and the report are regenerated with",
        align="justify",
    )
    paragraph(doc, "py -3.14 -m pip install -r requirements.txt", align="center", first_line=False, space_after=2)
    paragraph(doc, "py -3.14 experiments/run_experiments.py", align="center", first_line=False, space_after=2)
    paragraph(doc, "py -3.14 experiments/build_report.py", align="center", first_line=False, space_after=8)
    paragraph(
        doc,
        "Python 3.14 was used. On the machine where this was run, the python command points at a 3.15 preview that does not have NumPy installed, so the py -3.14 launcher is the reliable entry point.",
        align="justify",
    )

    heading(doc, "References")
    references = [
        "P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, “Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 7, pp. 711–720, 1997.",
        "R. A. Fisher, “The use of multiple measurements in taxonomic problems,” Annals of Eugenics, vol. 7, no. 2, pp. 179–188, 1936.",
        "A. M. Martínez and A. C. Kak, “PCA versus LDA,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 23, no. 2, pp. 228–233, 2001.",
        "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. Olivetti faces are loaded from sklearn.datasets.fetch_olivetti_faces.",
        "M. Turk and A. Pentland, “Eigenfaces for recognition,” Journal of Cognitive Neuroscience, vol. 3, no. 1, pp. 71–86, 1991.",
        "Q. Wang, Q. Gao, X. Gao, and F. Nie, “Two-dimensional PCA with F-norm minimization,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 31, no. 1, 2017.",
        "J. Yang, D. Zhang, A. F. Frangi, and J. Yang, “Two-dimensional PCA: A new approach to appearance-based face representation and recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 1, pp. 131–137, 2004.",
        "M.-X. Zhao, Z.-G. Jia, D.-W. Gong, and Y. Zhang, “Data-driven bilateral generalized two-dimensional quaternion principal component analysis with application to color face recognition,” arXiv:2306.07045, 2023. https://arxiv.org/abs/2306.07045",
        "J. Liu, M. Feng, X. Xiu, and W. Liu, “Towards robust and sparse linear discriminant analysis for image classification,” Pattern Recognition, vol. 153, 110512, 2024.",
        "J. Zhou, Q. Zhang, S. Zeng, B. Zhang, and L. Fang, “Latent linear discriminant analysis for feature extraction via isometric structural learning,” Pattern Recognition, vol. 149, 110218, 2024.",
    ]
    for item in references:
        paragraph(doc, item, size=11, align="left", space_after=6, first_line=False)

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
