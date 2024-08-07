// Fill out your copyright notice in the Description page of Project Settings.


#include "Pulley.h"
#include "EndEffector.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/Engine.h" 

// Sets default values
APulley::APulley()
{
    // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    Pulley_Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Pulley"));
    RootComponent = Pulley_Mesh;

    CableComponent = CreateDefaultSubobject<UCableComponent>(TEXT("CableComponent"));
    CableComponent->SetupAttachment(Pulley_Mesh);

    PhysicsConstraint = CreateDefaultSubobject<UPhysicsConstraintComponent>(TEXT("PhysicsConstraint"));
    PhysicsConstraint->SetupAttachment(Pulley_Mesh);

    MaxCableLength = 1172.0f; // Set a default maximum cable length
    MinCableLength = 0.0f;
    Tension = 1000.0f;
    MIN_Tension = 0.0f;
    MAX_Tension = 3000.0f;
    CableLength = 468.59f;
}

// Called when the game starts or when spawned
void APulley::BeginPlay()
{
	Super::BeginPlay();
    AttachCableToEndEffector();

}

// Called every frame
void APulley::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
    if (PIN_CONNECT) {
        if (!End_Effector->PhysicsMesh->IsSimulatingPhysics())
        {
            GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("Physics simulation not enabled on PhysicsMesh!"));
        }
        APulley::ApplyCableTension(CableLength, Tension);
    }
    
}

void APulley::AttachCableToEndEffector() {
    TArray<AActor*> Put_EndEffector;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AEndEffector::StaticClass(), Put_EndEffector);

    // Assuming we have found at least one instance of AMyActorB
    if (Put_EndEffector.Num() > 0)
    {
        End_Effector = Cast<AEndEffector>(Put_EndEffector[0]);
    }

    switch (Pulley_Number) {
    case 1: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_1, End_Effector->Pin_1->GetFName());
        Pin = End_Effector->Pin_1;
        break;
    case 2: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_2, End_Effector->Pin_2->GetFName());
        Pin = End_Effector->Pin_2;
        break;
    case 3: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_3, End_Effector->Pin_3->GetFName());
        Pin = End_Effector->Pin_3;
        break;
    case 4: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_4, End_Effector->Pin_4->GetFName());
        Pin = End_Effector->Pin_4;
        break;
    case 5: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_5, End_Effector->Pin_5->GetFName());
        Pin = End_Effector->Pin_5;
        break;
    case 6: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_6, End_Effector->Pin_6->GetFName());
        Pin = End_Effector->Pin_6;
        break;
    case 7: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_7, End_Effector->Pin_7->GetFName());
        Pin = End_Effector->Pin_7;
        break;
    case 8: 
        CableComponent->SetAttachEndToComponent(End_Effector->Pin_8, End_Effector->Pin_8->GetFName());
        Pin = End_Effector->Pin_8;
        break;
        
    default:
        PIN_CONNECT = false;
    }

    PIN_CONNECT = true;
}

void APulley::ApplyCableTension(float input_CableLength, float input_Tension)
{
    // CableComponent, Pin, End_Effector가 모두 유효한지 확인합니다.
    if (CableComponent && Pin && End_Effector)
    {
        // 케이블의 시작 위치를 가져옵니다.
        FVector StartLocation = CableComponent->GetComponentLocation();

        // 케이블의 끝 위치를 가져옵니다.
        FVector EndLocation = Pin->GetComponentLocation();

        // 시작 위치와 끝 위치 사이의 방향 벡터를 계산하고, 정규화합니다.
        FVector Direction = -1 * (EndLocation - StartLocation).GetSafeNormal();

        // 현재 케이블 길이를 계산합니다.
        float CurrentLength = (EndLocation - StartLocation).Size();

        // 현재 케이블 길이가 입력된 것보다 짧으면 힘을 받고 끌어당겨야 함
        if (CurrentLength > input_CableLength)
        {
            // 최소 케이블보단 커야 함
            float AdjustedDesiredLength = FMath::Max(input_CableLength, MinCableLength);

            // 조정된 케이블 길이에 맞춰 클램프된 끝 위치를 계산합니다.
            FVector ClampedEndLocation = StartLocation + Direction * AdjustedDesiredLength;

            // 엔드 이펙터가 이동할 방향을 정규화된 벡터로 계산합니다.
            FVector ForceDirection = (ClampedEndLocation - End_Effector->PhysicsMesh->GetComponentLocation()).GetSafeNormal();

            // 힘의 크기를 계산합니다.
            float ForceStrength = (ClampedEndLocation - End_Effector->PhysicsMesh->GetComponentLocation()).Size() * input_Tension;

            // 엔드 이펙터에 힘을 가합니다.
            End_Effector->PhysicsMesh->AddForceAtLocation(ForceDirection * ForceStrength, ClampedEndLocation);

            // 케이블 컴포넌트의 렌더 상태를 갱신합니다.
            CableComponent->MarkRenderStateDirty();
        }
        else if (CurrentLength == input_CableLength)
        {
            // 엔드 이펙터에 힘을 가합니다.
            End_Effector->PhysicsMesh->AddForceAtLocation(Direction * input_Tension, EndLocation);

            // 케이블 컴포넌트의 렌더 상태를 갱신합니다.
            CableComponent->MarkRenderStateDirty();
        }
        else
        {
            // 현재 케이블 길이가 입력된 것보다 길면 힘이 0이 됨
            End_Effector->PhysicsMesh->AddForceAtLocation(FVector(0, 0, 0), EndLocation);

            // 케이블 컴포넌트의 렌더 상태를 갱신합니다.
            CableComponent->MarkRenderStateDirty();
        }
    }
}

FVector APulley::InverseKinematics(const FVector& a, const FVector& b, const FVector& Location, const FVector& Rotation)
{
    float alpha = Rotation[0];
    float beta = Rotation[1];
    float gamma = Rotation[2];

    // Calculate the rotation matrix RA
    FMatrix RA = FMatrix::Identity;
    RA.M[0][0] = FMath::Cos(alpha) * FMath::Cos(beta);
    RA.M[0][1] = -FMath::Sin(alpha) * FMath::Cos(gamma) + FMath::Cos(alpha) * FMath::Sin(beta) * FMath::Sin(gamma);
    RA.M[0][2] = FMath::Sin(alpha) * FMath::Sin(gamma) + FMath::Cos(alpha) * FMath::Sin(beta) * FMath::Cos(gamma);

    RA.M[1][0] = FMath::Cos(alpha) * FMath::Sin(beta);
    RA.M[1][1] = FMath::Cos(alpha) * FMath::Cos(gamma) + FMath::Sin(alpha) * FMath::Sin(beta) * FMath::Sin(gamma);
    RA.M[1][2] = -FMath::Cos(alpha) * FMath::Sin(gamma) + FMath::Sin(alpha) * FMath::Sin(beta) * FMath::Cos(gamma);

    RA.M[2][0] = -FMath::Sin(beta);
    RA.M[2][1] = FMath::Cos(beta) * FMath::Sin(gamma);
    RA.M[2][2] = FMath::Cos(beta) * FMath::Cos(gamma);

    // Calculate the end position
    FVector end_position(Location[0], Location[1], Location[2]);

    // Compute the vector L
    FVector L = a - end_position - RA.TransformVector(b);

    // Calculate the length of the vector L
    return L;
}