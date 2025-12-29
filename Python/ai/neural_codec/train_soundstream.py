import torch
from torch.utils.data import DataLoader

from dataset import LibriSpeechDataset

from models import SoundStream
from models import STFTDiscriminator
from models.soundstream import loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoundStream().to(device)
discriminator = STFTDiscriminator().to(device)
optimizer_d = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)

# Create a DataLoader to handle batching of the training data
batch_size = 128
num_epochs = 20
dataset = LibriSpeechDataset(
    root_dir="data/LibriSpeech",
    subset="train-clean-100",
    sample_rate=24000,
    segment_seconds=1.0,
)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

λ_wave = 1.0
λ_stft = 1.0
λ_vq = 1.0
λ_adv = 0.0
λ_fm = 0.0

# Training loop
loss_history = []
for epoch in range(num_epochs):
    total_loss = 0.0

    for x in train_loader:
        x = x.to(device)


        # ====== Discriminator step ======
        with torch.no_grad():
            x_hat, _, _ = model(x)

        d_real, real_feats = discriminator(x)
        d_fake, fake_feats = discriminator(x_hat.detach())

        loss_d = discriminator_loss(d_real, d_fake)

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # ====== Generator step ======
        x_hat, _, vq_loss = model(x)

        d_fake, fake_feats = discriminator(x_hat)
        d_real, real_feats = discriminator(x)

        loss_wave = waveform_loss(x_hat, x)
        loss_stft = stft_loss(x_hat, x)
        loss_adv = generator_adversarial_loss(d_fake)
        loss_fm = feature_matching_loss(real_feats, fake_feats)

        loss_g = (
            λ_wave * loss_wave +
            λ_stft * loss_stft +
            λ_adv * loss_adv +
            λ_fm * loss_fm +
            λ_vq * vq_loss
        )

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # Update the running loss
        total_loss += loss.item()

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    #scheduler.step()
    loss_history.append(epoch_loss)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )
