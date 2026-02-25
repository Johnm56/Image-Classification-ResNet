ef train(args, data_loader,val_loader, model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        mixup_prob = 0.5  # Probability of applying MixUp
        alpha = 0.8       # MixUp hyperparameter
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        for inputs, labels in tqdm(data_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(args.device), labels.to(args.device)  # Ensure both are on same device
            r = np.random.rand()
            if r < mixup_prob:
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0), device=args.device)

                # Don't index inputs before moving to device
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = labels, labels[index]

                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()