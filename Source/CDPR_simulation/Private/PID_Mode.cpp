

/*
void APID_Mode::BeginPlay()
{
    Super::BeginPlay();
    Get_EndEffector() // 엔드이펙터 가져오기
}

void APID_Mode::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
}

void APID_Mode::Get_EndEffector() {
    TArray<AActor*> Put_EndEffector;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AEndEffector::StaticClass(), Put_EndEffector);

    // Assuming we have found at least one instance of AMyActorB
    if (Put_EndEffector.Num() > 0)
    {
        End_Effector = Cast<AEndEffector>(Put_EndEffector[0]);
    }
}

float APID_Mode::Inverse_Kinematics(const FVector& a, const FVector& b, const FVector& X) {
    // X는 [x, y, z, alpha, beta, gamma]를 포함합니다.
    float alpha = X.X;
    float beta = X.Y;
    float gamma = X.Z;

    // 회전 행렬 RA 계산
    FMatrix RA = FMatrix::Identity;
    RA.M[0][0] = FMath::Cos(alpha) * FMath::Cos(beta);
    RA.M[0][1] = -FMath::Sin(alpha) * FMath::Cos(gamma) + FMath::Cos(alpha) * FMath::Sin(beta) * FMath::Sin(gamma);
    RA.M[0][2] = FMath::Sin(alpha) * FMath::Sin(gamma) + FMath::Cos(alpha) * FMath::Sin(beta) * FMath::Cos(gamma);
    RA.M[1][0] = FMath::Sin(alpha) * FMath::Cos(beta);
    RA.M[1][1] = FMath::Cos(alpha) * FMath::Cos(gamma) + FMath::Sin(alpha) * FMath::Sin(beta) * FMath::Sin(gamma);
    RA.M[1][2] = -FMath::Cos(alpha) * FMath::Sin(gamma) + FMath::Cos(alpha) * FMath::Sin(beta) * FMath::Cos(gamma);
    RA.M[2][0] = -FMath::Sin(beta);
    RA.M[2][1] = FMath::Cos(beta) * FMath::Sin(gamma);
    RA.M[2][2] = FMath::Cos(beta) * FMath::Cos(gamma);

    // 엔드 이펙터의 위치
    FVector end_position = FVector(X.X, X.Y, X.Z);

    // a, b, end_position의 타입을 FVector에서 변환 (필요에 따라 수정)
    FVector aVec = FVector(a.X, a.Y, a.Z);
    FVector bVec = FVector(b.X, b.Y, b.Z);

    // L 벡터 계산
    FVector temp = aVec - end_position - RA.TransformVector(bVec);

    // lengths 계산
    return temp.Size();
}

FVector APID_Mode::Cal_Force() {
    FVector Force;
    for (TActorIterator<APulley> player(GetWorld()); player; ++player) {
        // player는 APulley* 형식입니다.
        APulley* Pulley = *player;

        // 케이블 1번 처리
        if (player->GetName().Contains(TEXT("BP_Pulley1"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 2번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley2"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 3번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley2"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 4번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley3"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 5번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley4"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 6번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley5"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 7번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley6"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        // 케이블 8번 처리
        else if (player->GetName().Contains(TEXT("BP_Pulley7"))) {
            Force += force_apply(Pulley, End_Effector);
        }
        else {
            continue;
        }

    }
    return Force;
}

FVector APID_Mode::force_apply(APulley* pulley, AEndEffector* e) {
    float Theory_lengths = Inverse_Kinematics(pulley->CableComponent->GetComponentLocation(), pulley->Pin->GetComponentLocation(), e->PhysicsMesh->GetComponentLocation());
    if (pulley->CableLength > Theory_lengths) { // 이론상보다 길다면
        return FVector(0, 0, 0); // 힘이 적용될리가 없지
    }
    else { // 짧다면 힘이 적용은 되겠지....
        // 짧다면 생각해야 할게 여러가지가 있다.
        // 1. 짧다면 케이블쪽으로 힘이 적용된다.
        // 2. 하지만 나머지 케이블들의 길이가 변함이 없다면 위치는 변하지 않는데 장력은 심하게 걸린다. 
        return pulley->ApplyCableTension();
    }
}

void APID_Mode::EndEffector_Force_Apply() {
    FVector total_force = Cal_Force();
    End_Effector->PhysicsMesh->Add
}
*/