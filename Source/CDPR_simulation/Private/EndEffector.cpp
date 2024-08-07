// Fill out your copyright notice in the Description page of Project Settings.


#include "EndEffector.h"

// Sets default values
AEndEffector::AEndEffector()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

    // 풀리 메시 설정
    PhysicsMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PhysicsMesh"));
    SetRootComponent(PhysicsMesh);

    static ConstructorHelpers::FObjectFinder<UStaticMesh> MeshAsset(TEXT("/Game/Cube"));
    if (MeshAsset.Succeeded())
    {
        PhysicsMesh->SetStaticMesh(MeshAsset.Object);
        PhysicsMesh->SetWorldLocation(FVector(0, 0, 0));
    }

    // 엔드이펙터 핀 설정
    Pin_1 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_1"));
    Pin_1->SetupAttachment(PhysicsMesh);
    Pin_1->SetWorldLocation(FVector(-40, 25, -10));

    Pin_2 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_2"));
    Pin_2->SetupAttachment(PhysicsMesh);
    Pin_2->SetWorldLocation(FVector(40, 25, -10));

    Pin_3 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_3"));
    Pin_3->SetupAttachment(PhysicsMesh);
    Pin_3->SetWorldLocation(FVector(40, -25, -10));

    Pin_4 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_4"));
    Pin_4->SetupAttachment(PhysicsMesh);
    Pin_4->SetWorldLocation(FVector(-40, -25, -10));

    Pin_5 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_5"));
    Pin_5->SetupAttachment(PhysicsMesh);
    Pin_5->SetWorldLocation(FVector(-40, 25, 10));

    Pin_6 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_6"));
    Pin_6->SetupAttachment(PhysicsMesh);
    Pin_6->SetWorldLocation(FVector(40, 25, 10));

    Pin_7 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_7"));
    Pin_7->SetupAttachment(PhysicsMesh);
    Pin_7->SetWorldLocation(FVector(40, -25, 10));

    Pin_8 = CreateDefaultSubobject<USceneComponent>(TEXT("Pin_8"));
    Pin_8->SetupAttachment(PhysicsMesh);
    Pin_8->SetWorldLocation(FVector(-40, -25, 10));
}
 

// Called when the game starts or when spawned
void AEndEffector::BeginPlay()
{
	Super::BeginPlay();
    // Set default static mesh (ensure you have a valid mesh)
    
}

// Called every frame
void AEndEffector::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

