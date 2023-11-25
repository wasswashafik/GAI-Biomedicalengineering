# GAI-Biomedicalengineering
Generative artificial intelligence (GAI) has emerged as a significant catalyst in the pitch of biomedical engineering (BME), besides deep learning and computer vision, bringing about profound changes in the domains of research, diagnostics, and therapies. This study succinctly summarizes the diverse effects and pivotal significance of GAI within this particular domain. It examines the wide range of applications in various fields, for example, drug discovery, artificial intelligence (AI)-enabled genomic data analysis, medical imaging, personalized medicine, and bioengineering. It further highlights the utilization of GAI-driven generative models to accelerate the process of drug development, improve the interpretation of medical images, enable customized treatment plans, and aid in the production of custom-made prosthetics. Nonetheless, this technical progression is not devoid of its obstacles. Matters like the quality of data, the interpretability of outcomes provided by AI, and ethical implications demand thorough examination to be scrutinized. However, within the context of these obstacles, there exists significant potential. The future holds great potential in the arena of neural network improvements, as well as collaborative efforts that bring together skills in AI and BME. The integration of GAI into clinical settings offers new avenues for exploration and development. This study presents a thorough examination that highlights the crucial significance of generative AI within the field of BME while also recognizing the intricate nature and ethical considerations associated with its implementation. Through an examination of these concepts, this study will establish a foundation for a comprehensive comprehension of the dynamic connection between AI innovation and medicinal progress since it has collected recently published literature of GAI, application, considerations, challenges and limitations, review articles, implementation and diffusion models.

# Citation
Please consider cite our paper if you find this repo is helpful. The link will be provieded after the publication


# Fresh Papers
- Sohail, S. S. (2023). A promising start and not a panacea: ChatGPT's early impact and potential in medical science and biomedical engineering research. Annals of Biomedical Engineering, 1-5.
- Mannuru, N. R., Shahriar, S., Teel, Z. A., Wang, T., Lund, B. D., Tijani, S., ... & Vaidya, P. (2023). Artificial intelligence in developing countries: The impact of generative artificial intelligence (AI) technologies for development. Information Development, 02666669231200628.
- Chen, T., Hong, L., Yudistyra, V., Vincoff, S., & Chatterjee, P. (2023). Generative design of therapeutics that bind and modulate protein states. Current Opinion in Biomedical Engineering, 100496.
- Huang, J., Neill, L., Wittbrodt, M., Melnick, D., Klug, M., Thompson, M., ... & Etemadi, M. (2023). Generative Artificial Intelligence for Chest Radiograph Interpretation in the Emergency Department. JAMA network open, 6(10), e2336100-e2336100.
- Jain, P., & Gupta, S. (2023). Blood flow prediction in multi-exposure speckle contrast imaging using conditional generative adversarial network. Cureus, 15(4).
- Cheng, K., Li, Z., He, Y., Guo, Q., Lu, Y., Gu, S., & Wu, H. (2023). Potential use of artificial intelligence in infectious disease: take ChatGPT as an example. Annals of Biomedical Engineering, 1-6.


## Contents

* [The Role Applications of GAIs in Biomedical Engineering](#applications-of-gans-in-agriculture)
  * [Application of GAI in Biomedical Engineering](#drug-discovery-development)
    * [Drug Discovery Development](#drug-discovery-development)
    * [Biomedical Imaging](#biomedical-imaging)
    * [Personalized Medicine](#personalized-medicine)
    * [Disease Modeling and Prediction](#disease-modeling-and-prediction)
    * [Drug Repurposing and Optimization](#drug-repurposing-and-optimization)
    * [Clinical Trial Optimization](#clinical-trial-optimization)
    * [Genomic Editing and Design](#genomic-editing-and-design)
    * [Drug Side Effects Prediction](#drug-side-effects-prediction)
  * [Ethical and Regulatory Considerations](#ethical-and-regulatory-considerations)
    * [Ethical GAI Consideration](ethical-GAI-consideration)
    * [Regulatory Consideration](#regulatory-consideration)
* [Generative Artificial Intelligence Challenges and Limitations](#generative-artificial-intelligence-challenges-and-limitations)
    * [Data Scarcity and Quality](#data-scarcity-and-quality)
    * [Regulatory and Ethical Constraints](regulatory-and-ethical-constraints)
    * [Model Explainability, Interpretability, and Transparency](#model-explainability-interpretability-and-transparency)
    * [Resource Intensiveness and Computational Complexity](#resource-intensiveness-and-computational-complexity)
    * [Bias and Generalization Issues](#bias-and-generalization-issues)
    * [Vulnerability to Adversarial Attacks](#vulnerability-to-adversarial-attacks)
* [GAI Review Papers](#gai-review-papers)
  
[GAI Implementations](#gan-implementations)
- Real Pytorch: https://realpython.com/generative-adversarial-networks/
- SaS: https://blogs.sas.com/content/sascom/2023/03/03/generative-ai-benefits-risks-and-a-framework-for-responsible-innovation/
- Google Cloud: https://cloud.google.com/ai/generative-ai
- Gartner Experts: https://www.gartner.com/en/topics/generative-ai

  * [Applications of Diffusion Models in Biomedical Engineering](#applications-of-diffusion-models-in-biomedical-engineering)

# The Role Applications of GAIs
**2023**
- Duffourc, M., & Gerke, S. (2023). Generative AI in health care and liability risks for physicians and safety concerns for patients. Jama. [[paper]](https://jamanetwork.com/journals/jama/article-abstract/2807168).
- Huang, J., Neill, L., Wittbrodt, M., Melnick, D., Klug, M., Thompson, M., ... & Etemadi, M. (2023). Generative Artificial Intelligence for Chest Radiograph Interpretation in the Emergency Department. JAMA network open, 6(10), e2336100-e2336100. [[Paper]](https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2810195).
- Lubowitz, J. H. (2023). Guidelines for the Use of Generative Artificial Intelligence Tools for Biomedical Journal Authors and Reviewers. Arthroscopy: The Journal of Arthroscopic & Related Surgery. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0749806323008812).
- Liu, Y., Yang, Z., Yu, Z., Liu, Z., Liu, D., Lin, H., ... & Shi, S. (2023). Generative artificial intelligence and its applications in materials science: Current situation and future perspectives. Journal of Materiomics. [[paper]](https://www.sciencedirect.com/science/article/pii/S2352847823000771).
- Mesko, B. (2023). The ChatGPT (Generative Artificial Intelligence) Revolution Has Made Artificial Intelligence Approachable for Medical Professionals. Journal of Medical Internet Research, 25, e48392. [[paper]](https://www.jmir.org/2023/1/e48392/)
- Al-Imam, A., Al-Hadithi, N., Alissa, F., & Michalak, M. (2023). Generative artificial intelligence in academic medical writing. Medical Journal of Babylon, 20(3), 654-656. [[paper]](https://journals.lww.com/mjby/_layouts/15/oaks.journals/downloadpdf.aspx?an=01216716-202320030-00039).


# GAI Implementations
- Real Pytorch: https://realpython.com/generative-adversarial-networks/
- SaS: https://blogs.sas.com/content/sascom/2023/03/03/generative-ai-benefits-risks-and-a-framework-for-responsible-innovation/
- Google Cloud: https://cloud.google.com/ai/generative-ai
- Gartner Experts: https://www.gartner.com/en/topics/generative-ai

# Applications of Diffusion Models in Biomedical Engineering
**2023**
- Akram, S., Athar, M., Saeed, K., Razia, A., Muhammad, T., & Alghamdi, H. A. (2023). Mathematical simulation of double diffusion convection on peristaltic pumping of Ellis nanofluid due to induced magnetic field in a non-uniform channel: Applications of magnetic nanoparticles in biomedical engineering. Journal of Magnetism and Magnetic Materials, 569, 170408. [[Link]](https://www.sciencedirect.com/science/article/pii/S0304885323000574).
