for lr in 8e-4 1e-5 2e-5 3e-5 5e-5 8e-5 1e-6; do
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval14  --gs_sheet=ABSA_py_linked --aspect=restaurants
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval14 --gs_sheet=ABSA_py_linked --aspect=laptops
    python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval15  --gs_sheet=ABSA_py_linked3 --aspect=restaurants
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval16  --gs_sheet=ABSA_py_linked3 --aspect=restaurants
done
for lr in 8e-4 1e-5 2e-5 3e-5 5e-5 8e-5 1e-6; do
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval14  --gs_sheet=ABSA_py_linked --aspect=restaurants
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval14 --gs_sheet=ABSA_py_linked --aspect=laptops
    # python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval15  --gs_sheet=ABSA_py_linked3 --aspect=restaurants
    python train_automodel2.py --batch_size=32 --lr=$lr --model=Bert --dataset=SemEval16  --gs_sheet=ABSA_py_linked3 --aspect=restaurants
done