// Fill out your copyright notice in the Description page of Project Settings.


#include "Pulley.h"
#include "EndEffector.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"

// Sets default values
APulley::APulley()
{
    // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    // Ǯ�� ���� ����, �������Ʈ���� �����ϵ��� ������
    Pulley_Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Pulley"));
    RootComponent = Pulley_Mesh;

    // Create Cable Component
    CableComponent = CreateDefaultSubobject<UCableComponent>(TEXT("CableComponent"));
    CableComponent->SetupAttachment(Pulley_Mesh);

    CableConstraint = CreateDefaultSubobject<UPhysicsConstraintComponent>(TEXT("CableConstraint"));
    CableConstraint->SetupAttachment(Pulley_Mesh);

    MaxCableLength = 1000.0f; // Set a default maximum cable length
    Tension = 1000.0f;

}

// Called when the game starts or when spawned
void APulley::BeginPlay()
{
	Super::BeginPlay();

    AttachCableToComponent();
	
}

// Called every frame
void APulley::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
    ApplyCableTension(Tension);
    LimitCableLength();
}

void APulley::AttachCableToComponent()
{
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AEndEffector::StaticClass(), FoundActors);

    if (FoundActors.Num() > 0)
    {
        EndEffector = Cast<AEndEffector>(FoundActors[0]);
        if (EndEffector && EndEffector->CablePin)
        {
            CableComponent->SetAttachEndTo(EndEffector, EndEffector->CablePin->GetFName());
        }
    }
}

void APulley::SetCableLength(float NewLength)
{
    if (CableComponent)
    {
        CableComponent->CableLength = FMath::Clamp(NewLength, 0.0f, MaxCableLength);
    }
}

void APulley::ApplyCableTension(float TensionStrength)
{
    if (CableComponent)
    {
        FVector StartLocation = CableComponent->GetComponentLocation();
        FVector EndLocation = CableComponent->GetComponentTransform().TransformPosition(CableComponent->EndLocation);
        FVector Direction = (EndLocation - StartLocation).GetSafeNormal();

        // ����� �ùķ��̼��ϱ� ���� ���̺� ������ ���� ���� ����
        Pulley_Mesh->AddForce(Direction * TensionStrength);

        // ���̺� ���� �ʰ� ���� Ȯ�� �� ���̺� ������ ó��
        float CurrentLength = (EndLocation - StartLocation).Size();
        if (CurrentLength > MaxCableLength)
        {
            // ���⼭ ���̺� ������ ���� ó��
            CableComponent->EndLocation = StartLocation + Direction * MaxCableLength;
        }
    }
}

void APulley::LimitCableLength() {
    if (CableComponent && EndEffector && EndEffector->CablePin)
    {
        FVector StartLocation = CableComponent->GetComponentLocation();
        FVector EndLocation = EndEffector->CablePin->GetComponentLocation();
        float CurrentLength = FVector::Dist(StartLocation, EndLocation);

        if (CurrentLength > MaxCableLength)
        {
            FVector ClampedLocation = StartLocation + (EndLocation - StartLocation).GetSafeNormal() * MaxCableLength;
            EndEffector->SetActorLocation(ClampedLocation);
        }
    }
}

void APulley::CutCable()
{
    if (CableComponent)
    {
        CableComponent->EndLocation = FVector::ZeroVector;
        CableComponent->SetAttachEndTo(nullptr, NAME_None);
    }
}