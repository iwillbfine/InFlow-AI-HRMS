-- 회사 테이블
INSERT INTO company (company_id, company_name, ceo, ceo_signature, business_registration_number, company_address, company_phone_number, company_stamp_url, company_logo_url)
VALUES
(1, '파도파도', '윤채연', 'https://inflow-company.s3.ap-northeast-2.amazonaws.com/ceo_signature.png', '229-81-30104', '서울 동작구 보라매로 87', '02-1234-5678', 'https://inflow-company.s3.ap-northeast-2.amazonaws.com/company_stamp.png', 'https://inflow-company.s3.ap-northeast-2.amazonaws.com/company_logo.png');

-- 근태상태유형 테이블
INSERT INTO attendance_status_type (attendance_status_type_code, attendance_status_type_name)
VALUES
    ('AS001', '정상출근'),
    ('AS002', '지각'),
    ('AS003', '조퇴'),
    ('AS004', '결근'),
    ('AS005', '휴가'),
    ('AS006', '병가'),
    ('AS007', '출장'),
    ('AS008', '재택근무'),
    ('AS009', '공가'),
    ('AS010', '연차'),
    ('AS011', '퇴근');

-- 직위 테이블
INSERT INTO `position` (position_code, position_name)
VALUES
    ('P001', '사원'),
    ('P002', '대리'),
    ('P003', '과장'),
    ('P004', '차장'),
    ('P005', '부장'),
    ('P006', '이사'),
    ('P007', '상무'),
    ('P008', '전무'),
    ('P009', '부사장'),
    ('P010', '사장');

-- 직책 테이블
INSERT INTO `role` (role_code, role_name)
VALUES
    ('R001', '팀원'),
    ('R002', '팀장'),
    ('R003', '프로젝트 매니저'),
    ('R004', '본부장'),
    ('R005', '부서장'),
    ('R006', '실장'),
    ('R007', '총괄'),
    ('R008', '대표'),
    ('R009', '감사'),
    ('R010', '고문');
    
-- 직무 테이블
INSERT INTO duty (duty_code, duty_name)
VALUES
    ('D001', '회계'),
    ('D002', '인사'),
    ('D003', '영업'),
    ('D004', '마케팅'),
    ('D005', '기술지원'),
    ('D006', '개발'),
    ('D007', '디자인'),
    ('D008', '기획'),
    ('D009', '데이터 분석'),
    ('D010', '품질 관리');

-- 부서 테이블
INSERT INTO department (department_code, department_name, created_at, disbanded_at, min_employee_num, upper_department_code)
VALUES
    ('DP001', '경영지원부', '2023-01-10 09:00:00', NULL, 5, NULL),
    ('DP002', '인사부', '2023-02-15 09:00:00', NULL, 3, 'DP001'),
    ('DP003', '재무부', '2023-03-01 09:00:00', NULL, 2, 'DP001'),
    ('DP004', '영업부', '2023-01-20 09:00:00', NULL, 10, NULL),
    ('DP005', '마케팅부', '2023-04-10 09:00:00', NULL, 6, NULL),
    ('DP006', 'IT기술지원부', '2023-05-15 09:00:00', NULL, 7, NULL),
    ('T001', '영업1팀', '2023-04-01 09:00:00', NULL, 4, 'DP004'),
    ('T002', '영업2팀', '2023-05-10 09:00:00', NULL, 4, 'DP004'),
    ('T003', '인사관리팀', '2023-06-01 09:00:00', NULL, 2, 'DP002'),
    ('T004', '재무회계팀', '2023-07-15 09:00:00', NULL, 3, 'DP003'),
    ('T005', '기획팀', '2023-08-01 09:00:00', NULL, 3, 'DP004'),
    ('T006', '기술지원팀', '2023-09-01 09:00:00', NULL, 4, 'DP006');

-- 부서구성원 테이블 트리거
DELIMITER //
CREATE TRIGGER after_employee_insert
    AFTER INSERT ON employee
    FOR EACH ROW
BEGIN
    DECLARE role_name VARCHAR(50) DEFAULT '팀원';
    DECLARE attendance_status_type_name VARCHAR(50) DEFAULT '정상출근';
    DECLARE target_department_code VARCHAR(20);

    -- position_code에 따라 role_name 설정
    SET role_name = CASE NEW.position_code
                        WHEN 'P001' THEN '팀원'
                        WHEN 'P002' THEN '팀장'
                        WHEN 'P003' THEN '프로젝트 매니저'
                        WHEN 'P004' THEN '본부장'
                        WHEN 'P005' THEN '부서장'
                        WHEN 'P006' THEN '실장'
                        WHEN 'P007' THEN '총괄'
                        WHEN 'P008' THEN '대표'
                        WHEN 'P009' THEN '감사'
                        WHEN 'P010' THEN '고문'
                        ELSE '팀원'
        END;

    -- attendance_status_type_code에 따라 attendance_status_type_name 설정
    SET attendance_status_type_name = CASE NEW.attendance_status_type_code
                                          WHEN 'AS001' THEN '정상출근'
                                          WHEN 'AS002' THEN '지각'
                                          WHEN 'AS003' THEN '조퇴'
                                          WHEN 'AS004' THEN '결근'
                                          WHEN 'AS005' THEN '휴가'
                                          WHEN 'AS006' THEN '병가'
                                          WHEN 'AS007' THEN '출장'
                                          WHEN 'AS008' THEN '재택근무'
                                          WHEN 'AS009' THEN '공가'
                                          WHEN 'AS010' THEN '연차'
                                          WHEN 'AS011' THEN '퇴근'
                                          ELSE '정상출근'
        END;

    -- 상위 부서에서 하위 부서를 결정 (임의 기준으로 결정)
    SET target_department_code = (
        SELECT d.department_code
        FROM department d
        WHERE d.upper_department_code = NEW.department_code
        LIMIT 1 -- 다수의 하위 부서가 있는 경우 임의로 하나를 선택
    );

    -- 하위 부서가 없는 경우, 기존 상위 부서를 그대로 사용
    IF target_department_code IS NULL THEN
        SET target_department_code = NEW.department_code;
    END IF;

    -- department_member 테이블에 새 사원 정보 추가
    INSERT INTO department_member (
        employee_number,
        name,
        role_name,
        email,
        profile_img_url,
        phone_number,
        attendance_status_type_name,
        manager_status,
        department_code,
        employee_id
    )
    VALUES (
               NEW.employee_number,
               NEW.name,
               role_name,
               NEW.email,
               NEW.profile_img_url,
               NEW.phone_number,
               attendance_status_type_name,
               CASE
                   WHEN NEW.position_code IN ('P002', 'P005', 'P004', 'P003') THEN 'Y'
                   ELSE 'N'
                   END,
               target_department_code,
               NEW.employee_id
           );
END//
DELIMITER ;


-- 사원 테이블
INSERT INTO employee (
    employee_id, employee_number, employee_role, password, gender, name, birth_date,
    email, phone_number, profile_img_url, join_date,
    join_type, resignation_date, resignation_status, salary, monthly_salary,
    street_address, detailed_address, postcode, department_code,
    attendance_status_type_code, position_code, role_code, duty_code
)

VALUES
    (1, '202000001', 'ADMIN', '$2a$10$gEF/iaV.jiHyAL0c8TZ2Aufen4ovoQyZX9ipoTKXUSIZ8h9XDlmFa', 'MALE', '홍길동', '1985-03-15',
     'hong@company.com', '010-1234-5678', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_admin.png', '2020-01-01 09:00:00',
     'ROOKIE', NULL, 'N', 50000000, 4000000, '서울 강남구 개포로 109길 5', '101동 101호', '06335',
     'DP001', 'AS001', 'P001', 'R001', 'D003'),
    (2, '202100001', 'EMPLOYEE', 'password456', 'FEMALE', '김영희', '1990-07-22',
     'kim@company.com', '010-2345-6789', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_basic_profile.png', '2021-03-10 09:00:00',
     'VETERAN', NULL, 'N', 55000000, 4500000, '서울 강남구 개포로 109길 9', '202동 202호', '06335',
     'DP002', 'AS002', 'P002', 'R002', 'D004'),
    (3, '201800001', 'EMPLOYEE', 'password789', 'MALE', '박철수', '1982-11-05',
     'park@company.com', '010-3456-7890', 'https://example.com/profile3.jpg', '2018-06-20 09:00:00',
     'VETERAN', '2024-05-30 18:00:00', 'Y', 70000000, 5000000, '서울 강남구 양재대로 478', '303동 303호', '06358',
     'DP003', 'AS004', 'P003', 'R003', 'D005'),
    (4, '202200001', 'EMPLOYEE', 'password101', 'FEMALE', '이수정', '1995-12-17',
     'lee@company.com', '010-4567-8901', 'https://example.com/profile4.jpg', '2022-02-25 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3800000, '서울 강남구 삼성로 11', '404동 404호', '06327',
     'DP004', 'AS003', 'P004', 'R004', 'D006'),
    (5, '202000002', 'EMPLOYEE', 'password102', 'MALE', '최강욱', '1992-09-30',
     'choi@company.com', '010-5678-9012', 'https://example.com/profile5.jpg', '2020-05-10 09:00:00',
     'ROOKIE', NULL, 'N', 48000000, 3900000, '서울시 강남구 개포로 416', '505동 505호', '06324',
     'DP005', 'AS002', 'P005', 'R005', 'D007'),
-- 인사팀 직원
     (6, '201900001', 'HR', '$2a$10$gEF/iaV.jiHyAL0c8TZ2Aufen4ovoQyZX9ipoTKXUSIZ8h9XDlmFa', 'FEMALE', '윤지혜', '1993-05-12',
     'yoon@company.com', '010-6789-0123', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_hr.png', '2019-04-01 09:00:00',
     'VETERAN', NULL, 'N', 47000000, 3900000, '서울 강남구 논현로 509', '606동 606호', '06349',
     'DP002', 'AS001', 'P001', 'R001', 'D002'),
-- 3년차 개발직 대리
    (7, '202101234', 'EMPLOYEE', '$2a$10$gEF/iaV.jiHyAL0c8TZ2Aufen4ovoQyZX9ipoTKXUSIZ8h9XDlmFa', 'MALE', '한상민', '1995-10-21',
     'han@company.com', '010-7890-1234', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_man.png', '2021-03-15 09:00:00',
     'VETERAN', NULL, 'N', 36000000, 3000000, '서울 강남구 도산대로 311', '707동 707호', '06351',
     'DP006', 'AS001', 'P002', 'R001', 'D005'),
-- IT 기술 지원 팀 부장
   (8, '201500001', 'MANAGER', '$2a$10$gEF/iaV.jiHyAL0c8TZ2Aufen4ovoQyZX9ipoTKXUSIZ8h9XDlmFa', 'MALE', '서진우', '1979-07-15',
     'seo@company.com', '010-8901-2345', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_manager.png', '2015-01-10 09:00:00',
     'VETERAN', NULL, 'N', 90000000, 7500000, '서울 강남구 압구정로 102', '808동 808호', '06353',
     'DP006', 'AS001', 'P005', 'R005', 'D006'),
-- 9번 ~ 26번 사원: IT기술지원부 사원 (경기도 지역 주소와 사번 설정 완료)

  	 (9, '202100002', 'EMPLOYEE', '$2a$10$gEF/iaV.jiHyAL0c8TZ2Aufen4ovoQyZX9ipoTKXUSIZ8h9XDlmFa', 'FEMALE', '장은희', '1995-01-22',
     'jang@company.com', '010-9012-3456', 'https://inflow-emp-profile.s3.ap-northeast-2.amazonaws.com/emp_woman.png', '2021-09-30 09:00:00',
     'ROOKIE', NULL, 'N', 45000000, 3700000, '경기도 수원시 영통구 광교로 55', '102동 202호', '16704',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (10, '202000003', 'EMPLOYEE', 'password107', 'MALE', '조우주', '1991-03-14',
     'leej@company.com', '010-2345-6780', 'https://example.com/profile10.jpg', '2020-02-10 09:00:00',
     'VETERAN', NULL, 'N', 47000000, 3900000, '경기도 성남시 분당구 백현로 97', '201동 1502호', '13518',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),
    (11, '202200002', 'EMPLOYEE', 'password108', 'FEMALE', '박하늘', '1996-06-20',
     'parkh@company.com', '010-3456-7892', 'https://example.com/profile11.jpg', '2022-07-25 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 고양시 일산동구 정발산로 24', '302동 302호', '10405',
     'DP006', 'AS003', 'P001', 'R001', 'D005'),
    (12, '202100003', 'EMPLOYEE', 'password109', 'MALE', '김민수', '1993-08-30',
     'kimmin@company.com', '010-4567-8903', 'https://example.com/profile12.jpg', '2021-04-20 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 용인시 기흥구 흥덕2로 123', '402동 402호', '16950',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (13, '202100004', 'EMPLOYEE', 'password110', 'FEMALE', '최은정', '1994-02-11',
     'choeun@company.com', '010-5678-9014', 'https://example.com/profile13.jpg', '2021-05-15 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 부천시 부천로 50', '302동 505호', '14556',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (14, '202200003', 'EMPLOYEE', 'password111', 'MALE', '신동엽', '1992-12-20',
     'shin@company.com', '010-6789-0125', 'https://example.com/profile14.jpg', '2022-01-10 09:00:00',
     'ROOKIE', NULL, 'N', 47000000, 3800000, '경기도 평택시 평택로 155', '101동 1201호', '17747',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),
    (15, '202200004', 'EMPLOYEE', 'password112', 'FEMALE', '정윤아', '1995-07-18',
     'jung@company.com', '010-7890-1236', 'https://example.com/profile15.jpg', '2022-03-25 09:00:00',
     'ROOKIE', NULL, 'N', 45000000, 3700000, '경기도 안양시 동안구 관악대로 141', '101동 703호', '13931',
     'DP006', 'AS003', 'P001', 'R001', 'D005'),
    (16, '201900002', 'EMPLOYEE', 'password113', 'MALE', '오지현', '1989-04-02',
     'ohji@company.com', '010-8901-2347', 'https://example.com/profile16.jpg', '2019-07-01 09:00:00',
     'VETERAN', NULL, 'N', 48000000, 3800000, '경기도 남양주시 화도읍 마석로 56', '205동 808호', '12224',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (17, '202200005', 'EMPLOYEE', 'password114', 'FEMALE', '박수현', '1996-10-05',
     'parksh123@company.com', '010-9012-3458', 'https://example.com/profile17.jpg', '2022-04-12 09:00:00',
     'ROOKIE', NULL, 'N', 47000000, 3900000, '경기도 의정부시 시민로 23', '103동 1103호', '11652',
     'DP006', 'AS003', 'P001', 'R001', 'D005'),
    (18, '202000004', 'EMPLOYEE', 'password115', 'MALE', '이하늘', '1992-11-30',
     'leehn@company.com', '010-2345-6789', 'https://example.com/profile18.jpg', '2020-08-01 09:00:00',
     'VETERAN', NULL, 'N', 46000000, 3800000, '경기도 파주시 교하로 240', '505동 505호', '10932',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),
    (19, '202200006', 'EMPLOYEE', 'password116', 'FEMALE', '김은서', '1998-01-12',
     'kimse@company.com', '010-3456-7890', 'https://example.com/profile19.jpg', '2022-09-15 09:00:00',
     'ROOKIE', NULL, 'N', 45000000, 3700000, '경기도 광명시 광명로 121', '101동 303호', '14305',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (20, '202100005', 'EMPLOYEE', 'password117', 'MALE', '정민호', '1993-03-03',
     'jungmh@company.com', '010-5678-9011', 'https://example.com/profile20.jpg', '2021-06-25 09:00:00',
     'ROOKIE', NULL, 'N', 47000000, 3900000, '경기도 하남시 미사대로 250', '301동 1004호', '12918',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),

-- 21번 ~ 26번 사원: IT기술지원부 사원 (경기도 지역 주소와 사번 설정 완료)
    (21, '202110006', 'EMPLOYEE', 'password118', 'FEMALE', '윤채은', '1997-04-25',
     'yoonce@company.com', '010-6789-0122', 'https://example.com/profile21.jpg', '2021-10-01 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 김포시 김포한강3로 210', '502동 1502호', '10003',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (22, '202010005', 'EMPLOYEE', 'password119', 'MALE', '차정훈', '1990-09-11',
     'chajh@company.com', '010-7890-1233', 'https://example.com/profile22.jpg', '2020-12-15 09:00:00',
     'VETERAN', NULL, 'N', 48000000, 3800000, '경기도 군포시 산본로 200', '305동 1003호', '15820',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (23, '202100007', 'EMPLOYEE', 'password120', 'FEMALE', '이선영', '1994-07-30',
     'leese@company.com', '010-8901-2344', 'https://example.com/profile23.jpg', '2021-05-20 09:00:00',
     'ROOKIE', NULL, 'N', 47000000, 3800000, '경기도 의왕시 철도박물관로 112', '101동 503호', '16071',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),
    (24, '201910003', 'EMPLOYEE', 'password121', 'MALE', '김재환', '1988-02-17',
     'kimjh123@company.com', '010-3456-7891', 'https://example.com/profile24.jpg', '2019-08-25 09:00:00',
     'VETERAN', NULL, 'N', 49000000, 3800000, '경기도 양주시 고덕로 150', '202동 902호', '11487',
     'DP006', 'AS001', 'P001', 'R001', 'D005'),
    (25, '202200007', 'EMPLOYEE', 'password122', 'FEMALE', '박수영', '1995-06-05',
     'parksy@company.com', '010-4567-8902', 'https://example.com/profile25.jpg', '2022-03-18 09:00:00',
     'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 구리시 건원대로 230', '303동 1305호', '11915',
     'DP006', 'AS003', 'P001', 'R001', 'D005'),
    (26, '202000006', 'EMPLOYEE', 'password123', 'MALE', '최용준', '1989-10-27',
     'choiyj123@company.com', '010-5678-9013', 'https://example.com/profile26.jpg', '2020-06-12 09:00:00',
     'VETERAN', NULL, 'N', 47000000, 3900000, '경기도 포천시 소흘읍 송우리 345', '701동 104호', '11135',
     'DP006', 'AS002', 'P001', 'R001', 'D005'),

-- 경기도 한정 27번~50번
	(27, '202200008', 'EMPLOYEE', 'password124', 'FEMALE', '정가은', '1995-11-18',
	'jungge@company.com', '010-6789-1234', 'https://example.com/profile27.jpg', '2022-01-15 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '경기도 안산시 단원구 중앙대로 100', '101동 1203호', '15357',
	'DP003', 'AS001', 'P001', 'R001', 'D003'),
	(28, '202100008', 'EMPLOYEE', 'password125', 'MALE', '김도훈', '1993-04-12',
	'kimdh@company.com', '010-7890-1235', 'https://example.com/profile28.jpg', '2021-04-01 09:00:00',
	'VETERAN', NULL, 'N', 48000000, 3800000, '경기도 화성시 동탄대로 50', '102동 904호', '18590',
	'DP004', 'AS002', 'P002', 'R002', 'D004'),
	(29, '202000007', 'EMPLOYEE', 'password126', 'FEMALE', '최현아', '1987-08-15',
	'choiha@company.com', '010-1234-5679', 'https://example.com/profile29.jpg', '2020-06-20 09:00:00',
	'VETERAN', NULL, 'N', 52000000, 4000000, '경기도 남양주시 와부읍 덕소로 150', '203동 1202호', '12276',
	'DP002', 'AS001', 'P003', 'R003', 'D002'),
	(30, '202100009', 'EMPLOYEE', 'password127', 'MALE', '이준석', '1994-10-02',
	'leejs@company.com', '010-2345-6781', 'https://example.com/profile30.jpg', '2021-09-25 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '경기도 평택시 서재로 230', '501동 1103호', '17714',
	'DP001', 'AS001', 'P001', 'R001', 'D001'),
	(31, '202100010', 'EMPLOYEE', 'password128', 'FEMALE', '문하영', '1996-07-18',
	'moonhy@company.com', '010-3456-7892', 'https://example.com/profile31.jpg', '2021-05-30 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 광주시 역동로 89', '701동 503호', '12780',
	'DP005', 'AS001', 'P002', 'R001', 'D008'),
	(32, '202100011', 'EMPLOYEE', 'password129', 'MALE', '강성민', '1992-09-27',
	'kangsm@company.com', '010-4567-8903', 'https://example.com/profile32.jpg', '2021-03-10 09:00:00',
	'VETERAN', NULL, 'N', 49000000, 3900000, '경기도 고양시 덕양구 행신로 45', '203동 1102호', '10594',
	'DP004', 'AS002', 'P001', 'R002', 'D009'),
	(33, '202100012', 'EMPLOYEE', 'password130', 'FEMALE', '한예은', '1998-06-11',
	'hanyee@company.com', '010-5678-9014', 'https://example.com/profile33.jpg', '2021-06-01 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 하남시 미사대로 200', '401동 1204호', '12918',
	'DP001', 'AS001', 'P001', 'R001', 'D005'),
	(34, '202000008', 'EMPLOYEE', 'password131', 'MALE', '윤지혁', '1990-12-05',
	'yoonjh@company.com', '010-6789-0125', 'https://example.com/profile34.jpg', '2020-11-12 09:00:00',
	'VETERAN', NULL, 'N', 53000000, 4000000, '경기도 용인시 기흥구 흥덕로 200', '302동 1001호', '16950',
	'DP003', 'AS003', 'P002', 'R003', 'D003'),
	(35, '202100013', 'EMPLOYEE', 'password132', 'FEMALE', '강미정', '1995-03-23',
	'kangmj@company.com', '010-7890-1236', 'https://example.com/profile35.jpg', '2021-02-15 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 파주시 와석로 78', '201동 805호', '10925',
	'DP005', 'AS001', 'P001', 'R001', 'D008'),
	(36, '202200009', 'EMPLOYEE', 'password133', 'MALE', '박상진', '1993-08-19',
	'parksj232@company.com', '010-8901-2347', 'https://example.com/profile36.jpg', '2022-03-05 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '경기도 안양시 동안구 관악대로 145', '103동 1202호', '13929',
	'DP003', 'AS001', 'P003', 'R001', 'D005'),
	(37, '202000009', 'EMPLOYEE', 'password134', 'FEMALE', '오지민', '1994-01-29',
	'ohjm@company.com', '010-9012-3458', 'https://example.com/profile37.jpg', '2020-09-10 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3900000, '경기도 시흥시 배곧로 100', '205동 1404호', '15010',
	'DP002', 'AS003', 'P001', 'R002', 'D006'),
	(38, '202200010', 'EMPLOYEE', 'password135', 'MALE', '김현우', '1991-07-13',
	'kimhw@company.com', '010-2345-6789', 'https://example.com/profile38.jpg', '2022-01-20 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 양평군 양근로 245', '201동 506호', '12507',
	'DP001', 'AS001', 'P001', 'R001', 'D001'),
	(39, '202100014', 'EMPLOYEE', 'password136', 'FEMALE', '박지수', '1995-11-30',
	'parksj@company.com', '010-3456-7893', 'https://example.com/profile39.jpg', '2021-05-21 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 광명시 오리로 90', '104동 803호', '14320',
	'DP004', 'AS002', 'P002', 'R002', 'D007'),
	(40, '202200011', 'EMPLOYEE', 'password137', 'MALE', '서은호', '1998-02-18',
	'seoho@company.com', '010-4567-8904', 'https://example.com/profile40.jpg', '2022-06-12 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 오산시 경기대로 220', '501동 1106호', '18129',
	'DP005', 'AS001', 'P001', 'R001', 'D008'),
	(41, '202000010', 'EMPLOYEE', 'password138', 'FEMALE', '나민정', '1996-05-22',
	'namj@company.com', '010-5678-9015', 'https://example.com/profile41.jpg', '2020-12-03 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3800000, '경기도 의정부시 시민로 80', '601동 1407호', '11650',
	'DP003', 'AS002', 'P003', 'R002', 'D009'),
	(42, '202100015', 'EMPLOYEE', 'password139', 'MALE', '조현수', '1997-10-11',
	'johs@company.com', '010-6789-0126', 'https://example.com/profile42.jpg', '2021-11-20 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3900000, '경기도 광주시 성남로 250', '102동 703호', '12765',
	'DP004', 'AS001', 'P002', 'R001', 'D003'),
	(43, '202100016', 'EMPLOYEE', 'password140', 'FEMALE', '김나리', '1994-06-08',
	'kimnr@company.com', '010-7890-1237', 'https://example.com/profile43.jpg', '2021-08-15 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '경기도 안산시 상록구 사동로 40', '305동 506호', '15324',
	'DP005', 'AS003', 'P001', 'R001', 'D004'),
	(44, '202100017', 'EMPLOYEE', 'password141', 'MALE', '이동훈', '1993-02-04',
	'leodh@company.com', '010-8901-2348', 'https://example.com/profile44.jpg', '2021-02-10 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3900000, '경기도 고양시 일산동구 백마로 150', '202동 1402호', '10425',
	'DP001', 'AS001', 'P002', 'R002', 'D006'),
	(45, '202200012', 'EMPLOYEE', 'password142', 'FEMALE', '최유정', '1998-09-05',
	'choiyj@company.com', '010-1234-5678', 'https://example.com/profile45.jpg', '2022-03-19 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 시흥시 은행로 99', '203동 1006호', '15027',
	'DP004', 'AS002', 'P003', 'R003', 'D005'),
	(46, '202100018', 'EMPLOYEE', 'password143', 'MALE', '박준영', '1992-08-24',
	'parkjy123@company.com', '010-3456-7895', 'https://example.com/profile46.jpg', '2021-09-30 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '경기도 성남시 수정구 남문로 40', '101동 1208호', '13101',
	'DP003', 'AS001', 'P001', 'R002', 'D008'),
	(47, '202200013', 'EMPLOYEE', 'password144', 'FEMALE', '김혜진', '1997-05-21',
	'kimhj@company.com', '010-4567-8906', 'https://example.com/profile47.jpg', '2022-05-18 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 군포시 군포로 160', '102동 1001호', '15855',
	'DP001', 'AS003', 'P001', 'R001', 'D003'),
	(48, '202100019', 'EMPLOYEE', 'password145', 'MALE', '이태우', '1989-01-28',
	'leetw@company.com', '010-5678-9017', 'https://example.com/profile48.jpg', '2021-04-22 09:00:00',
	'VETERAN', NULL, 'N', 49000000, 3900000, '경기도 안성시 금광로 121', '304동 1205호', '17539',
	'DP005', 'AS002', 'P002', 'R002', 'D004'),
	(49, '202200014', 'EMPLOYEE', 'password146', 'FEMALE', '조혜진', '1996-04-02',
	'johj@company.com', '010-6789-0128', 'https://example.com/profile49.jpg', '2022-06-30 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '경기도 화성시 봉담읍 와우로 40', '205동 503호', '18397',
	'DP002', 'AS001', 'P003', 'R001', 'D006'),
	(50, '202200015', 'EMPLOYEE', 'password147', 'MALE', '한세준', '1993-11-29',
	'hansj@company.com', '010-7890-1239', 'https://example.com/profile50.jpg', '2022-07-10 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '경기도 포천시 이동면 경기동로 90', '701동 1203호', '11130',
	'DP005', 'AS002', 'P001', 'R001', 'D005'),

-- 51번 ~ 65번 사원: 인천광역시 주소와 DP006 제외 부서 설정
	(51, '202200016', 'EMPLOYEE', 'password148', 'FEMALE', '이민아', '1995-09-22',
	'leema@company.com', '010-6789-1230', 'https://example.com/profile51.jpg', '2022-01-25 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '인천광역시 남동구 구월로 30', '101동 401호', '21574',
	'DP001', 'AS001', 'P001', 'R001', 'D002'),
	(52, '202100020', 'EMPLOYEE', 'password149', 'MALE', '조태호', '1992-05-10',
	'joth@company.com', '010-7890-1231', 'https://example.com/profile52.jpg', '2021-02-14 09:00:00',
	'VETERAN', NULL, 'N', 49000000, 3800000, '인천광역시 서구 청라대로 100', '202동 1102호', '22741',
	'DP003', 'AS002', 'P002', 'R002', 'D003'),
	(53, '202100021', 'EMPLOYEE', 'password150', 'FEMALE', '박유진', '1993-06-18',
	'parkjy@company.com', '010-1234-5671', 'https://example.com/profile53.jpg', '2021-09-05 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '인천광역시 부평구 경원대로 55', '302동 202호', '21405',
	'DP005', 'AS003', 'P001', 'R001', 'D004'),
	(54, '202100022', 'EMPLOYEE', 'password151', 'MALE', '김민호', '1991-07-23',
	'kimmh@company.com', '010-2345-6782', 'https://example.com/profile54.jpg', '2021-03-12 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3900000, '인천광역시 계양구 계양대로 170', '401동 804호', '21063',
	'DP002', 'AS001', 'P002', 'R002', 'D005'),
	(55, '202200017', 'EMPLOYEE', 'password152', 'FEMALE', '정혜인', '1997-12-04',
	'jungji@company.com', '010-3456-7894', 'https://example.com/profile55.jpg', '2022-06-25 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '인천광역시 남동구 논현로 120', '105동 505호', '21684',
	'DP001', 'AS001', 'P001', 'R001', 'D003'),
	(56, '202100023', 'EMPLOYEE', 'password153', 'MALE', '서지훈', '1989-08-15',
	'seojh@company.com', '010-4567-8905', 'https://example.com/profile56.jpg', '2021-11-10 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '인천광역시 연수구 송도과학로 85', '103동 1202호', '21984',
	'DP004', 'AS002', 'P001', 'R002', 'D006'),
	(57, '202100024', 'EMPLOYEE', 'password154', 'FEMALE', '김연주', '1994-09-02',
	'kimyj@company.com', '010-5678-9016', 'https://example.com/profile57.jpg', '2021-07-01 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '인천광역시 중구 운남로 30', '302동 506호', '22386',
	'DP005', 'AS001', 'P001', 'R001', 'D008'),
	(58, '202200018', 'EMPLOYEE', 'password155', 'MALE', '장우진', '1995-11-13',
	'jangwj@company.com', '010-6789-0123', 'https://example.com/profile58.jpg', '2022-02-17 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '인천광역시 동구 샛골로 75', '203동 1103호', '22551',
	'DP003', 'AS002', 'P003', 'R003', 'D009'),
	(59, '202100025', 'EMPLOYEE', 'password156', 'FEMALE', '윤소희', '1996-03-25',
	'yoonsh@company.com', '010-7890-1234', 'https://example.com/profile59.jpg', '2021-12-18 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '인천광역시 미추홀구 미추홀대로 210', '104동 307호', '22231',
	'DP001', 'AS003', 'P001', 'R001', 'D005'),
	(60, '202100026', 'EMPLOYEE', 'password157', 'MALE', '한준수', '1992-07-17',
	'hanjs@company.com', '010-1234-5672', 'https://example.com/profile60.jpg', '2021-08-10 09:00:00',
	'ROOKIE', NULL, 'N', 49000000, 3800000, '인천광역시 부평구 부평대로 70', '105동 907호', '21422',
	'DP004', 'AS001', 'P002', 'R001', 'D006'),
	(61, '202200019', 'EMPLOYEE', 'password158', 'FEMALE', '신혜수', '1993-10-15',
	'shinhs@company.com', '010-2345-6783', 'https://example.com/profile61.jpg', '2022-03-22 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '인천광역시 연수구 인천타워대로 150', '103동 304호', '21932',
	'DP005', 'AS003', 'P002', 'R001', 'D007'),
	(62, '202200020', 'EMPLOYEE', 'password159', 'MALE', '유성민', '1998-11-22',
	'yusm@company.com', '010-3456-7895', 'https://example.com/profile62.jpg', '2022-05-12 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '인천광역시 계양구 계양대로 100', '101동 804호', '21057',
	'DP003', 'AS001', 'P002', 'R001', 'D001'),
	(63, '202200021', 'EMPLOYEE', 'password160', 'FEMALE', '조하영', '1994-06-11',
	'johy@company.com', '010-4567-8906', 'https://example.com/profile63.jpg', '2022-07-30 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '인천광역시 서구 원당로 65', '201동 1205호', '22768',
	'DP001', 'AS001', 'P001', 'R001', 'D008'),
	(64, '202200022', 'EMPLOYEE', 'password161', 'MALE', '고재훈', '1991-02-18',
	'gojh@company.com', '010-5678-9017', 'https://example.com/profile64.jpg', '2022-02-02 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '인천광역시 동구 솔빛로 100', '301동 907호', '22535',
	'DP004', 'AS003', 'P003', 'R001', 'D009'),
	(65, '202100027', 'EMPLOYEE', 'password162', 'FEMALE', '배윤정', '1992-09-01',
	'baeyj@company.com', '010-6789-0128', 'https://example.com/profile65.jpg', '2021-10-15 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '인천광역시 남구 매소홀로 50', '305동 1303호', '22150',
	'DP002', 'AS002', 'P001', 'R002', 'D005'),

-- 66번 ~ 90번 사원: 서울특별시 주소와 DP006 제외 부서 설정
	 (66, '202200023', 'EMPLOYEE', 'password163', 'MALE', '장민수', '1993-04-22',
	'jangms@company.com', '010-1234-5678', 'https://example.com/profile66.jpg', '2022-03-15 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 강남구 테헤란로 150', '101동 202호', '06236',
	'DP001', 'AS001', 'P001', 'R001', 'D002'),
	(67, '202100028', 'EMPLOYEE', 'password164', 'FEMALE', '김소현', '1994-08-14',
	'kimsh@company.com', '010-2345-6781', 'https://example.com/profile67.jpg', '2021-05-11 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 서초구 서초대로 60', '203동 603호', '06595',
	'DP003', 'AS001', 'P002', 'R002', 'D003'),
	(68, '202100029', 'EMPLOYEE', 'password165', 'MALE', '이정훈', '1992-03-25',
	'leejh@company.com', '010-3456-7892', 'https://example.com/profile68.jpg', '2021-07-19 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '서울특별시 용산구 한남대로 80', '402동 1502호', '04401',
	'DP004', 'AS002', 'P002', 'R003', 'D004'),
	(69, '202100030', 'EMPLOYEE', 'password166', 'FEMALE', '박수빈', '1995-01-12',
	'parksb@company.com', '010-4567-8903', 'https://example.com/profile69.jpg', '2021-09-10 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 동작구 흑석로 40', '105동 1101호', '06901',
	'DP002', 'AS001', 'P003', 'R002', 'D005'),
	(70, '202200024', 'EMPLOYEE', 'password167', 'MALE', '신재영', '1993-06-07',
	'shinjy@company.com', '010-5678-9014', 'https://example.com/profile70.jpg', '2022-04-10 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '서울특별시 성동구 왕십리로 220', '201동 1005호', '04794',
	'DP005', 'AS002', 'P001', 'R001', 'D007'),
	(71, '202100031', 'EMPLOYEE', 'password168', 'FEMALE', '차은지', '1990-02-11',
	'chaeji@company.com', '010-6789-0125', 'https://example.com/profile71.jpg', '2021-08-01 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3900000, '서울특별시 중구 소공로 70', '301동 204호', '04535',
	'DP003', 'AS001', 'P001', 'R001', 'D009'),
	(72, '202100032', 'EMPLOYEE', 'password169', 'MALE', '오세민', '1996-11-17',
	'osemin@company.com', '010-7890-1236', 'https://example.com/profile72.jpg', '2021-03-18 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 강서구 화곡로 45', '201동 702호', '07628',
	'DP001', 'AS002', 'P002', 'R001', 'D008'),
	(73, '202200025', 'EMPLOYEE', 'password170', 'FEMALE', '고지원', '1997-08-29',
	'gojw@company.com', '010-1234-5677', 'https://example.com/profile73.jpg', '2022-02-20 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '서울특별시 강북구 도봉로 110', '202동 903호', '01032',
	'DP002', 'AS001', 'P001', 'R001', 'D005'),
	(74, '202200026', 'EMPLOYEE', 'password171', 'MALE', '백준호', '1993-09-21',
	'baekjh@company.com', '010-2345-6789', 'https://example.com/profile74.jpg', '2022-05-25 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 송파구 송파대로 400', '104동 805호', '05613',
	'DP005', 'AS003', 'P003', 'R002', 'D004'),
	(75, '202100033', 'EMPLOYEE', 'password172', 'FEMALE', '윤다인', '1998-12-06',
	'yundi@company.com', '010-3456-7894', 'https://example.com/profile75.jpg', '2021-06-15 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 관악구 남부순환로 290', '303동 303호', '08810',
	'DP004', 'AS002', 'P001', 'R003', 'D006'),
	(76, '202200027', 'EMPLOYEE', 'password173', 'MALE', '조현석', '1991-07-29',
	'jochs@company.com', '010-4567-8906', 'https://example.com/profile76.jpg', '2022-08-10 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 은평구 응암로 160', '401동 1404호', '03477',
	'DP003', 'AS001', 'P002', 'R001', 'D001'),
	(77, '202100034', 'EMPLOYEE', 'password174', 'FEMALE', '신가현', '1994-05-15',
	'shingh@company.com', '010-5678-9018', 'https://example.com/profile77.jpg', '2021-05-28 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 종로구 종로대로 120', '101동 1205호', '03195',
	'DP001', 'AS001', 'P001', 'R001', 'D003'),
	(78, '202200028', 'EMPLOYEE', 'password175', 'MALE', '박세훈', '1992-09-30',
	'parksh@company.com', '010-6789-0126', 'https://example.com/profile78.jpg', '2022-09-05 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 성북구 화랑로 300', '501동 703호', '02798',
	'DP005', 'AS001', 'P003', 'R001', 'D009'),
	(79, '202200029', 'EMPLOYEE', 'password176', 'FEMALE', '김현수', '1993-03-01',
	'kimhs@company.com', '010-7890-1238', 'https://example.com/profile79.jpg', '2022-07-12 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '서울특별시 금천구 가산디지털로 100', '201동 1303호', '08590',
	'DP004', 'AS003', 'P001', 'R001', 'D008'),
	(80, '202100035', 'EMPLOYEE', 'password177', 'MALE', '정하나', '1996-04-07',
	'jungha@company.com', '010-1234-5679', 'https://example.com/profile80.jpg', '2021-04-15 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '서울특별시 마포구 마포대로 200', '401동 203호', '04168',
	'DP001', 'AS002', 'P003', 'R002', 'D004'),
	(81, '202100036', 'EMPLOYEE', 'password178', 'FEMALE', '박서준', '1997-06-30',
	'parksej@company.com', '010-2345-6780', 'https://example.com/profile81.jpg', '2021-10-05 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 노원구 동일로 320', '203동 905호', '01850',
	'DP003', 'AS001', 'P001', 'R001', 'D007'),
	(82, '202200030', 'EMPLOYEE', 'password179', 'MALE', '유수빈', '1990-10-21',
	'yusb@company.com', '010-3456-7895', 'https://example.com/profile82.jpg', '2022-04-22 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '서울특별시 서대문구 연희로 150', '105동 1301호', '03730',
	'DP005', 'AS001', 'P001', 'R002', 'D006'),
	(83, '202100037', 'EMPLOYEE', 'password180', 'FEMALE', '최다해', '1998-03-09',
	'choidh@company.com', '010-4567-8909', 'https://example.com/profile83.jpg', '2021-11-22 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 양천구 목동로 220', '202동 1003호', '07995',
	'DP002', 'AS003', 'P001', 'R001', 'D008'),
	(84, '202100038', 'EMPLOYEE', 'password181', 'MALE', '한지훈', '1994-01-15',
	'hanjh@company.com', '010-5678-9019', 'https://example.com/profile84.jpg', '2021-03-03 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 강동구 천호대로 300', '101동 501호', '05399',
	'DP001', 'AS001', 'P002', 'R002', 'D003'),
	(85, '202200031', 'EMPLOYEE', 'password182', 'FEMALE', '이해나', '1993-12-23',
	'leeha@company.com', '010-6789-0127', 'https://example.com/profile85.jpg', '2022-02-17 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '서울특별시 중랑구 망우로 210', '103동 204호', '02100',
	'DP003', 'AS001', 'P003', 'R001', 'D005'),
	(86, '202100039', 'EMPLOYEE', 'password183', 'MALE', '김정환', '1997-07-01',
	'kimjh@company.com', '010-7890-1239', 'https://example.com/profile86.jpg', '2021-09-14 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '서울특별시 구로구 디지털로 300', '204동 502호', '08378',
	'DP005', 'AS003', 'P001', 'R001', 'D009'),
	(87, '202200032', 'EMPLOYEE', 'password184', 'FEMALE', '정지윤', '1994-05-14',
	'jungjy@company.com', '010-1234-5678', 'https://example.com/profile87.jpg', '2022-05-30 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '서울특별시 동대문구 장안로 100', '304동 604호', '02630',
	'DP004', 'AS001', 'P002', 'R003', 'D006'),
	(88, '202200033', 'EMPLOYEE', 'password185', 'MALE', '차현우', '1995-10-16',
	'chahw@company.com', '010-2345-6789', 'https://example.com/profile88.jpg', '2022-06-14 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '서울특별시 종로구 삼청로 50', '102동 1407호', '03018',
	'DP003', 'AS003', 'P003', 'R001', 'D007'),
	(89, '202100040', 'EMPLOYEE', 'password186', 'FEMALE', '한미주', '1992-06-10',
	'hanmj@company.com', '010-3456-7890', 'https://example.com/profile89.jpg', '2021-04-08 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3700000, '서울특별시 광진구 아차산로 100', '105동 903호', '05025',
	'DP002', 'AS001', 'P001', 'R002', 'D001'),
	(90, '202200034', 'EMPLOYEE', 'password187', 'MALE', '이서준', '1991-09-05',
	'leej@company.com', '010-5678-9011', 'https://example.com/profile90.jpg', '2022-04-11 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 강남구 언주로 250', '104동 803호', '06108',
	'DP005', 'AS002', 'P002', 'R003', 'D004'),
	
-- 91번 ~ 100번 사원: 임의의 서울, 인천, 경기 주소와 DP006 제외 부서 설정

	(91, '202100041', 'EMPLOYEE', 'password188', 'MALE', '최상준', '1996-03-21',
	'choisj@company.com', '010-6789-1231', 'https://example.com/profile91.jpg', '2021-06-12 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 강북구 번동로 30', '101동 401호', '01010',
	'DP004', 'AS001', 'P001', 'R001', 'D002'),
	(92, '202200035', 'EMPLOYEE', 'password189', 'FEMALE', '김유리', '1995-10-12',
	'kimyr@company.com', '010-7890-1232', 'https://example.com/profile92.jpg', '2022-08-18 09:00:00',
	'ROOKIE', NULL, 'N', 48000000, 3800000, '경기도 고양시 덕양구 고양대로 120', '202동 1002호', '10586',
	'DP002', 'AS003', 'P001', 'R001', 'D003'),
	(93, '202200036', 'EMPLOYEE', 'password190', 'MALE', '박찬영', '1993-08-10',
	'parkcy@company.com', '010-1234-5673', 'https://example.com/profile93.jpg', '2022-09-05 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 은평구 갈현로 80', '203동 903호', '03401',
	'DP003', 'AS001', 'P002', 'R003', 'D004'),
	(94, '202100042', 'EMPLOYEE', 'password191', 'FEMALE', '송은지', '1998-04-05',
	'songej@company.com', '010-2345-6784', 'https://example.com/profile94.jpg', '2021-11-15 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '인천광역시 계양구 봉오대로 100', '304동 1403호', '21023',
	'DP005', 'AS002', 'P001', 'R001', 'D008'),
	(95, '202100043', 'EMPLOYEE', 'password192', 'MALE', '고영호', '1991-01-30',
	'goyh@company.com', '010-3456-7896', 'https://example.com/profile95.jpg', '2021-07-10 09:00:00',
	'VETERAN', NULL, 'N', 47000000, 3900000, '경기도 부천시 길주로 150', '205동 302호', '14510',
	'DP001', 'AS001', 'P003', 'R002', 'D005'),
	(96, '202200037', 'EMPLOYEE', 'password193', 'FEMALE', '백나경', '1994-09-25',
	'baeknk@company.com', '010-4567-8907', 'https://example.com/profile96.jpg', '2022-02-12 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3700000, '서울특별시 동작구 흑석로 75', '101동 504호', '06923',
	'DP003', 'AS001', 'P001', 'R001', 'D009'),
	(97, '202100044', 'EMPLOYEE', 'password194', 'MALE', '이준호', '1997-07-15',
	'leejh@company.com', '010-5678-9018', 'https://example.com/profile97.jpg', '2021-10-01 09:00:00',
	'ROOKIE', NULL, 'N', 46000000, 3800000, '서울특별시 서초구 반포대로 120', '301동 205호', '06591',
	'DP005', 'AS002', 'P002', 'R003', 'D001'),
	(98, '202100045', 'EMPLOYEE', 'password195', 'FEMALE', '한예진', '1996-02-17',
	'hanyj@company.com', '010-6789-0129', 'https://example.com/profile98.jpg', '2021-03-25 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '서울특별시 마포구 대흥로 200', '403동 903호', '04141',
	'DP001', 'AS002', 'P002', 'R002', 'D007'),
	(99, '202100046', 'EMPLOYEE', 'password196', 'MALE', '최지호', '1993-11-05',
	'choijh@company.com', '010-1234-5680', 'https://example.com/profile99.jpg', '2021-06-22 09:00:00',
	'ROOKIE', NULL, 'N', 47000000, 3800000, '인천광역시 남동구 구월로 50', '205동 1304호', '21550',
	'DP004', 'AS001', 'P003', 'R001', 'D006'),
	(100, '202100047', 'EMPLOYEE', 'password197', 'FEMALE', '유진아', '1995-06-30',
	'yujina@company.com', '010-3456-7891', 'https://example.com/profile100.jpg', '2021-07-28 09:00:00',
	'ROOKIE', NULL, 'N', 45000000, 3700000, '경기도 수원시 영통구 광교로 200', '102동 1004호', '16503',
	'DP002', 'AS003', 'P001', 'R001', 'D008');


-- 가구원관계 테이블
INSERT INTO family_relationship (family_relationship_code, family_relationship_name)
VALUES
    ('SELF', '본인'),
    ('SPOUSE', '배우자'),
    ('CHILD', '자녀'),
    ('PARENT', '부모'),
    ('SIBLING', '형제자매'),
    ('GRANDPARENT', '조부모');

-- 가족구성원 테이블
INSERT INTO family_member (family_member_id, name, birth_date, employee_id, family_relationship_code)
VALUES
-- employee_id 1
(1, '홍길동', '1980-01-01', 1, 'SELF'),     -- 본인
(2, '배수지', '1986-06-15', 1, 'SPOUSE'),    -- 배우자
(3, '홍예지', '2015-08-20', 1, 'CHILD'),     -- 자녀

-- employee_id 2
(4, '김영수', '1975-02-15', 2, 'SELF'),      -- 본인
(5, '김갑수', '1965-02-25', 2, 'PARENT'),    -- 부모
(6, '전소민', '2018-01-10', 2, 'CHILD'),     -- 자녀

-- employee_id 3
(7, '박철수', '1960-07-11', 3, 'SELF'),      -- 본인
(8, '박해진', '1960-11-11', 3, 'PARENT'),    -- 부모
(9, '박철민', '1987-04-05', 3, 'SIBLING'),   -- 형제자매

-- employee_id 4
(10, '남궁희', '1994-07-30', 4, 'SELF'),     -- 본인
(11, '남궁민수', '1994-07-30', 4, 'SPOUSE'), -- 배우자
(12, '남궁빈', '2020-12-18', 4, 'CHILD'),    -- 자녀

-- employee_id 5
(13, '최영희', '1968-03-22', 5, 'SELF'),     -- 본인
(14, '최수진', '1968-03-22', 5, 'PARENT'),   -- 부모
(15, '최강민', '1990-09-12', 5, 'SIBLING');  -- 형제자매


-- 학력 테이블
INSERT INTO education (education_id, school_name, admission_date, graduation_date, degree, major, employee_id)
VALUES
    (1, '서울대학교', '2005-03-01 09:00:00', '2009-02-28 09:00:00', '학사', '컴퓨터공학', 1),
    (2, '연세대학교', '2006-03-01 09:00:00', '2010-02-28 09:00:00', '학사', '경영학', 2),
    (3, '고려대학교', '2007-03-01 09:00:00', '2011-02-28 09:00:00', '학사', '전자공학', 3),
    (4, 'KAIST', '2010-03-01 09:00:00', '2014-02-28 09:00:00', '학사', '기계공학', 4),
    (5, 'POSTECH', '2008-03-01 09:00:00', '2012-02-28 09:00:00', '학사', '화학공학', 5),
    (6, '서울대학교', '2004-03-01 09:00:00', '2008-02-28 09:00:00', '학사', '경제학', 1),
    (7, '연세대학교', '2009-03-01 09:00:00', '2013-02-28 09:00:00', '석사', '경영학', 2),
    (8, '고려대학교', '2012-03-01 09:00:00', '2016-02-28 09:00:00', '석사', '전자공학', 3),
    (9, 'KAIST', '2014-03-01 09:00:00', '2018-02-28 09:00:00', '석사', '기계공학', 4),
    (10, 'POSTECH', '2011-03-01 09:00:00', '2015-02-28 09:00:00', '석사', '화학공학', 5);

-- 경력 테이블
INSERT INTO career (career_id, company_name, role_name, join_date, resignation_date, employee_id)
VALUES
    (1, '삼성전자', '소프트웨어 엔지니어', '2010-05-01 09:00:00', '2014-08-31 18:00:00', 1),
    (2, 'LG전자', '마케팅 매니저', '2012-03-01 09:00:00', '2016-12-31 18:00:00', 2),
    (3, 'SK텔레콤', '네트워크 엔지니어', '2013-06-01 09:00:00', '2017-05-31 18:00:00', 3),
    (4, '네이버', '프론트엔드 개발자', '2014-02-01 09:00:00', '2018-07-31 18:00:00', 4),
    (5, '카카오', '빅데이터 분석가', '2015-08-01 09:00:00', '2019-11-30 18:00:00', 5),
    (6, '삼성전자', '하드웨어 엔지니어', '2011-04-01 09:00:00', '2015-03-31 18:00:00', 1),
    (7, 'LG전자', '경영 기획', '2013-07-01 09:00:00', '2017-06-30 18:00:00', 2),
    (8, 'SK텔레콤', 'IT 기획자', '2014-09-01 09:00:00', '2018-08-31 18:00:00', 3),
    (9, '네이버', 'UX/UI 디자이너', '2015-05-01 09:00:00', '2019-04-30 18:00:00', 4),
    (10, '카카오', '모바일 앱 개발자', '2016-03-01 09:00:00', '2020-06-30 18:00:00', 5);

-- 계약서 table
INSERT INTO contract (contract_id, contract_type, created_at, file_name, file_url, contract_status, consent_status, employee_id)
VALUES 
(1, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 1),
(2, 'SECURITY', NULL, NULL, NULL, 'SIGNING', 'N', 1),
(3, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 2),
(4, 'SECURITY', '2024-02-02 09:00:00', '이순신_비밀유지서약서', 'https://example.com/contract4.pdf', 'REGISTERED', 'Y', 2),
(5, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 3),
(6, 'SECURITY', '2024-03-01 09:00:00', '강감찬_비밀유지서약서', 'https://example.com/contract6.pdf', 'REGISTERED', 'Y', 3),
(7, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 4),
(8, 'SECURITY', '2024-04-01 09:00:00', '홍길동_비밀유지서약서', 'https://example.com/contract8.pdf', 'REGISTERED', 'Y', 4),
(9, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 5),
(10, 'SECURITY', '2024-05-01 09:00:00', '박영희_비밀유지서약서', 'https://example.com/contract10.pdf', 'REGISTERED', 'Y', 5),
(11, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 6),
(12, 'SECURITY', '2024-06-01 09:00:00', '최재형_비밀유지서약서', 'https://example.com/contract12.pdf', 'REGISTERED', 'Y', 6),
(13, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 7),
(14, 'SECURITY', '2024-07-01 09:00:00', '정두영_비밀유지서약서', 'https://example.com/contract14.pdf', 'REGISTERED', 'Y', 7),
(15, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 8),
(16, 'SECURITY', '2024-08-01 09:00:00', '김철수_비밀유지서약서', 'https://example.com/contract16.pdf', 'REGISTERED', 'Y', 8),
(17, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 9),
(18, 'SECURITY', '2024-09-01 09:00:00', '박지민_비밀유지서약서', 'https://example.com/contract18.pdf', 'REGISTERED', 'Y', 9),
(19, 'EMPLOYMENT', NULL, NULL, NULL, 'SIGNING', 'N', 10),
(20, 'SECURITY', '2024-10-01 09:00:00', '김태희_비밀유지서약서', 'https://example.com/contract20.pdf', 'REGISTERED', 'Y', 10);

-- 자격증 테이블
INSERT INTO qualification (qualification_id, qualification_name, qualification_number, qualified_at, issuer, grade_score, employee_id)
VALUES
    (1, '정보처리기사', '123456789', '2023-12-01 09:00:00', '한국산업인력공단', 'PASS', 1),
    (2, '컴퓨터활용능력 1급', '987654321', '2024-01-15 09:00:00', '한국산업인력공단', 'PASS', 2),
    (3, '한국사능력검정시험 1급', '223344556', '2024-02-20 09:00:00', '국사편찬위원회', 'PASS', 4),
    (4, '회계관리 1급', '334455667', '2023-11-10 09:00:00', '한국세무사회', 'PASS', 5),
    (5, 'ERP 정보관리사', '445566778', '2024-03-05 09:00:00', '한국경영기술연구원', 'PASS', 1),
    (6, '운전면허 1종 보통', '556677889', '2023-08-01 09:00:00', '도로교통공단', 'PASS', 2),
    (7, 'SQLD', '667788990', '2024-04-10 09:00:00', '한국정보통신기술협회', 'PASS', 3),
    (8, '미용사', '778899001', '2023-07-15 09:00:00', '한국미용사회', 'PASS', 4),
    (9, '국가공인 자격증', '889900112', '2024-05-30 09:00:00', '국가자격관리원', 'PASS', 5);

-- 언어 테이블
INSERT INTO language (language_code, language_name)
VALUES
    ('EN', 'English'),
    ('JP', 'Japanese'),
    ('CN', 'Chinese'),
    ('FR', 'French'),
    ('DE', 'German');

-- 어학시험 테이블
INSERT INTO language_test (language_test_id, language_test_name, qualification_number, issuer, qualified_at, grade_score, employee_id, language_code)
VALUES
    (1, 'TOEIC', '123456789', 'ETS', '2023-11-01 09:00:00', '550', 1, 'EN'),
    (2, 'JLPT N1', '987654321', 'JLPT', '2024-02-10 09:00:00', 'PASS', 2, 'JP'),
    (3, 'HSK 5급', '223344556', 'HSK', '2023-12-05 09:00:00', 'PASS', 3, 'CN'),
    (4, 'DELF B2', '334455667', 'CIEP', '2024-01-15 09:00:00', 'PASS', 4, 'FR'),
    (5, 'TestDaF', '445566778', 'Goethe-Institut', '2024-03-01 09:00:00', 'C1', 5, 'DE');

-- 징계 및 포상 테이블
INSERT INTO discipline_reward (discipline_reward_id, discipline_reward_name, content, created_at, employee_id)
VALUES
    (1, '포상', '우수사원으로 선정되어 상금 100만원 지급', '2023-12-10 09:00:00', 1),
    (2, '징계', '업무 중 상사의 지시 불이행으로 경고처리', '2024-01-05 09:00:00', 2),
    (3, '포상', '해외출장 우수 성과를 달성하여 포상 휴가 제공', '2024-02-20 09:00:00', 3),
    (4, '징계', '직장 내 폭언으로 경고조치', '2024-03-01 09:00:00', 4),
    (5, '포상', '고객 만족도 우수 평가로 인센티브 지급', '2024-04-10 09:00:00', 5),
    (6, '징계', '무단결근으로 인한 감봉 조치', '2024-05-15 09:00:00', 1),
    (7, '포상', '팀 프로젝트 우수성 인정으로 팀원에게 상장 수여', '2024-06-20 09:00:00', 2),
    (8, '징계', '연차 미사용으로 경고처리', '2024-07-25 09:00:00', 3),
    (9, '포상', '특별한 기여로 사내 표창장 수여', '2024-08-30 09:00:00', 4),
    (10, '징계', '지각 반복으로 경고조치', '2024-09-10 09:00:00', 5);

-- 인사발령 항목  테이블
INSERT INTO appointment_item (appointment_item_code, appointment_item_name) VALUES
('PROM', '승진'),
('SPPR', '특진'),
('RETI', '퇴직'),
('DEMO', '강등'),
('TRNS', '부서이동'),
('RCHG', '보직변경');

-- 인사발령 테이블
INSERT INTO appointment (appointment_id, appointed_at, employee_id,authorizer_id, appointment_item_code, department_code, duty_code, role_code, position_code)
VALUES
-- 한상민의 첫 발령: 입사 시 주임으로 IT 기술 부서로 발령
(1, '2021-03-15 09:00:00', 7, 1, 'TRNS', 'DP006', 'D005', 'R001', 'P001'), -- 주임(P001)으로 입사

-- 한상민의 두 번째 발령: 대리로 승진
(2, '2024-03-15 09:00:00', 7, 1, 'PROM', 'DP006', 'D005', 'R001', 'P002'); -- 대리(P002)로 승진

-- 휴가유형 테이블
INSERT INTO vacation_type (vacation_type_id, vacation_type_name)
VALUES
    (1, '연차'),
    (2, '공가'),
    (3, '병가'),
    (4, '포상휴가'),
    (5, '특별휴가');

-- 휴가 정책 테이블
INSERT INTO vacation_policy (vacation_policy_id, vacation_policy_name, vacation_policy_description, vacation_policy_status, allocation_days, paid_status, year, created_at, auto_allocation_cycle, vacation_type_id, policy_register_id)
VALUES
    (1, '2024 1년 이상 근속자 연차', '2024년 연차는 1년 이상 근속한 사원에게 15일씩 자동 지급됩니다.', 'NORMAL', 15, 'Y', 2024, '2024-01-01 00:00:00', '0 0 0 1 1 *', 1, 5),
    (2, '2024 공가', '공가는 증빙자료가 있어야 지급 가능합니다.', 'NORMAL', 366, 'Y', 2024, '2024-01-01 00:00:00', '0 0 0 1 1 *', 2, 5),
    (3, '2024 병가', '병가는 진료확인서가 있어야 지급 가능합니다.', 'NORMAL', 240, 'Y', 2024, '2024-01-01 00:00:00', '0 0 0 1 1 *', 3, 5),
    (4, '2024 포상휴가', '2024년 포상휴가는 최대 3일까지 지급 가능합니다.', 'NORMAL', 3, 'Y', 2024, '2024-01-01 00:00:00', NULL, 4, 5),
    (5, '2024 특별휴가', '2024년 특별휴가는 최대 7일까지 지급 가능합니다.', 'NORMAL', 7, 'N', 2024, '2024-01-01 00:00:00', NULL, 5, 5),
    (6, '2024 1년 미만 근속자 연차', '2024년 연차는 1년 미만 근속한 사원에게 매월 1일씩 자동 지급됩니다.', 'ROOKIE', 1, 'Y', 2024, '2024-01-01 00:00:00', '0 0 0 1 * *', 1, 5);


-- 휴가 테이블
INSERT INTO vacation (
    vacation_id, vacation_name, vacation_left, vacation_used, created_at, expired_at, expiration_status,
    employee_id, vacation_policy_id, vacation_type_id
) VALUES
      (1, '2024년 연차', 15, 5, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 1, 1, 1),
      (2, '2024년 연차', 14, 1, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 2, 1, 1),
      (3, '2024년 병가', 10, 2, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 3, 2, 3),
      (4, '2024년 포상휴가', 7, 0, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 1, 4, 4),
      (5, '2024년 특별휴가', 5, 1, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 5, 5, 5),
      (6, '2024년 연차', 15, 3, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 3, 1, 1),
      (7, '2024년 공가', 3, 1, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 2, 3, 2),
      (8, '2024년 연차', 20, 4, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 4, 1, 1),
      (9, '2024년 연차', 10, 2, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 5, 1, 1),
      (10, '2024년 특별휴가', 5, 0, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 5, 5, 5),
      (11, '2024년 공가', 366, 0, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 1, 2, 2),
      (12, '2024년 병가', 240, 0, '2024-01-01 09:00:00', '2024-12-31 23:59:59', 'N', 1, 3, 3);

-- 휴가 신청 테이블
INSERT INTO vacation_request (
    vacation_request_id, start_date, end_date, created_at, request_reason,
    request_status, rejection_reason, canceled_at, cancel_reason, cancel_status,
    employee_id, vacation_id
)
VALUES
    (1, '2024-12-01 09:00:00', '2024-12-03 18:00:00', '2024-11-01 10:00:00', '휴가를 가고 싶습니다.',
     'REJECT', '휴가 사유 재작성 요망', NULL, NULL, 'N', 1, 1),
    (2, '2024-12-10 09:00:00', '2024-12-12 18:00:00', '2024-11-05 11:30:00', '가족 모임 참석',
     'WAIT', NULL, NULL, NULL, 'N', 2, 2),
    (3, '2024-12-20 09:00:00', '2024-12-22 18:00:00', '2024-11-10 14:00:00', '긴급한 일로 휴가 요청',
     'REJECT', '일정이 맞지 않아서', NULL, NULL, 'N', 3, 3),
    (4, '2024-12-25 09:00:00', '2024-12-27 18:00:00', '2024-11-15 16:00:00', '포상 휴가 사용',
     'WAIT', NULL, NULL, NULL, 'N', 1, 4),
    (5, '2024-12-30 09:00:00', '2024-12-31 18:00:00', '2024-11-20 08:00:00', '휴가',
     'WAIT', NULL, NULL, NULL, 'N', 5, 5),
    (6, '2025-01-01 09:00:00', '2025-01-03 18:00:00', '2024-11-30 10:00:00', '연차 사용',
     'ACCEPT', NULL, NULL, NULL, 'N', 1, 1);

-- 휴가 신청 증빙자료 테이블
INSERT INTO vacation_request_file (
    vacation_request_file_id, file_name, file_url, vacation_request_id
)
VALUES
    (1, '홍길동_휴가_증빙자료', 'https://example.com/files/vacation_request_1.pdf', 4),
    (2, '김영희_휴가_증빙자료', 'https://example.com/files/vacation_request_2.pdf', 2),
    (3, '박철수_휴가_증빙자료', 'https://example.com/files/vacation_request_3.pdf', 3),
    (4, '홍길동_휴가_증빙자료', 'https://example.com/files/vacation_request_4.pdf', 4),
    (5, '최강욱_휴가_증빙자료', 'https://example.com/files/vacation_request_5.pdf', 5);

-- 연차사용촉진정책 테이블
INSERT INTO annual_vacation_promotion_policy (
    annual_vacation_promotion_policy_id, month, day, standard
)
VALUES
    (1, 7, 1, 15),
    (2, 10, 1, 10);

-- 근태 신청 유형 테이블
INSERT INTO attendance_request_type (
    attendance_request_type_id, attendance_request_type_name, attendance_request_type_description
)
VALUES
    (1, '재택근무', '선재택 후승인'),
    (2, '초과근무', '초과근무는 30분 간격으로 연장이 가능합니다.'),
    (3, '출장', '출장을 가시려면 출장전에 출장 신청하시고, 신청 승인받으셔야 합니다.'),
    (4, '파견', '파견을 가시려면 파견 전에 파견 신청하시고, 신청 승인받으셔야 합니다.'),
    (5, '휴직', '휴직을 하려면 휴직 신청 후, 신청 승인받으셔야 합니다'),
    (6, '복직', '복직을 하려면 복직 신청을 하셔야 되는데, 신청에 복직 신청서 파일을 같이 첨부하셔야 합니다.');

-- 근태 신청 테이블
INSERT INTO attendance_request (
    attendance_request_id, request_reason, start_date, end_date, created_at,
    rejection_reason, request_status, canceled_at, cancel_reason, cancel_status,
    destination, employee_id, attendance_request_type_id
)
VALUES
    (1, '출장으로 인한 근무지 변경', '2024-11-10 09:00:00', '2024-11-10 18:00:00', '2024-11-01 09:00:00',
     NULL, 'WAIT', NULL, NULL, 'N', '서울시 강남구', 1, 3),
    (2, '재택근무 요청', '2024-11-05 09:00:00', '2024-11-05 18:00:00', '2024-11-02 09:30:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', NULL, 5, 1),
    (3, '초과근무 요청', '2024-11-07 19:00:00', '2024-11-07 22:00:00', '2024-11-03 10:00:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', NULL, 3, 2),
    (4, '파견 근무 요청', '2024-11-12 09:00:00', '2024-12-12 18:00:00', '2024-11-04 08:00:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', '부산 사무소', 4, 4),
    (5, '휴직 신청', '2024-11-02 00:00:00', '2030-12-30 23:59:59', '2024-11-06 09:15:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', NULL, 1, 5),
    (6, '한화시스템 부트캠프 멘토링', '2024-11-12 18:00:00', '2024-11-12 22:00:00', '2024-11-12 17:00:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', NULL, 1, 2),
    (7, '야근', '2024-12-01 18:00:00', '2024-12-01 20:00:00', '2024-12-01 12:00:00',
     NULL, 'ACCEPT', NULL, NULL, 'N', NULL, 1, 2);

-- 근태 신청 증빙자료 테이블
INSERT INTO attendance_request_file (
    attendance_request_file_id, file_name, file_url, attendance_request_id
)
VALUES
    (1, '근태 증빙자료1', 'https://example.com/file1.pdf', 1),
    (2, '근태 증빙자료2', 'https://example.com/file2.pdf', 2),
    (3, '근태 증빙자료3', 'https://example.com/file3.pdf', 3),
    (4, '근태 증빙자료4', 'https://example.com/file4.pdf', 4),
    (5, '근태 증빙자료5', 'https://example.com/file5.pdf', 5);

-- 출퇴근 테이블
INSERT INTO commute (
    commute_id, start_time, end_time, remote_status, overtime_status, employee_id, attendance_request_id
)
VALUES
    (1, '2024-11-01 09:00:00', '2024-11-01 18:00:00', 'N', 'N', 1, NULL),
    (2, '2024-11-01 08:30:00', '2024-11-01 17:30:00', 'N', 'N', 2, NULL),
    (3, '2024-11-02 09:15:00', '2024-11-02 18:15:00', 'N', 'Y', 3, 3),
    (4, '2024-11-02 09:00:00', '2024-11-02 18:00:00', 'N', 'N', 4, NULL),
    (5, '2024-11-03 09:00:00', '2024-11-03 17:00:00', 'Y', 'N', 5, 2),
    (6, '2024-11-02 09:00:00', '2024-11-02 18:00:00', 'Y', 'N', 1, NULL),
    (7, '2024-11-03 09:15:00', '2024-11-03 18:00:00', 'N', 'N', 1, NULL),
    (8, '2024-11-03 18:00:00', '2024-11-03 21:00:00', 'N', 'Y', 1, NULL),
    (9, '2024-11-04 09:00:00', '2024-11-04 18:00:00', 'N', 'N', 1, NULL),
    (10, '2024-11-05 09:00:00', '2024-11-05 18:00:00', 'N', 'N', 1, NULL),
    (11, '2024-11-06 09:00:00', '2024-11-06 18:00:00', 'N', 'N', 1, NULL),
    (12, '2024-11-07 09:00:00', '2024-11-07 18:00:00', 'N', 'N', 1, NULL),
    (13, '2024-11-08 09:00:00', '2024-11-08 18:00:00', 'N', 'N', 1, NULL),
    (14, '2024-11-09 09:00:00', '2024-11-09 18:00:00', 'N', 'N', 1, NULL),
    (15, '2024-11-10 09:00:00', '2024-11-10 18:00:00', 'N', 'N', 1, NULL),
    (16, '2024-11-11 09:00:00', '2024-11-11 18:00:00', 'N', 'N', 1, NULL),
    (17, '2024-11-12 09:00:00', '2024-11-12 18:00:00', 'N', 'N', 1, NULL),
    (18, '2024-11-12 18:00:00', '2024-11-12 22:00:00', 'N', 'Y', 1, 6),
    (19, '2024-12-01 09:00:00', '2024-12-01 18:00:00', 'N', 'N', 1, NULL),
    (20, '2024-12-01 18:00:00', '2024-12-01 20:00:00', 'N', 'Y', 1, 7);

-- 휴복직 테이블
INSERT INTO leave_return (
    leave_return_id, start_date, end_date, employee_id, attendance_request_id
)
VALUES
    (1, '2024-11-02 00:00:00', '2030-12-30 23:59:59', 1, 5);

-- 출장파견 테이블
INSERT INTO business_trip (
    business_trip_id, start_date, end_date, trip_type, destination, employee_id, attendance_request_id
)
VALUES
    (1, '2024-11-12 09:00:00', '2024-12-12 18:00:00', 'DISPATCH', '부산 사무소', 4, 4);

-- 과제유형 테이블


INSERT INTO task_type (task_type_id, task_type_name)
VALUES
    (1, '부서과제'),
    (2, '개인과제'),
    (3, '공통과제');

INSERT INTO evaluation_policy (
    evaluation_policy_id, start_date, end_date, year, half, task_ratio,
    min_rel_eval_count, created_at, modifiable_date, policy_description,
    policy_register_id, task_type_id
) VALUES
-- 2023년 상반기 부서별 과제 평가 정책
(1, '2023-06-01', '2023-07-30', 2023, '1st', 0.4,
 4, '2023-05-01', '2023-05-31',
 '부서별 평가 최종 환산 점수: 2023년 상반기 부서별 평가는 사원의 최종 평가 점수에 40%의 비율로 반영됩니다. 과제 반영 비율 제한: 부서별 평가의 총 비율은 40%로 설정되어 있으며, 각 과제별로 10% 단위로 비율이 부여될 수 있습니다. 즉, 각 과제는 4%, 8%, 12%, ... 40% 범위 내에서 비율이 설정되며, 총 비율인 40%를 초과할 수 없습니다. 과제별 최소 점수: 부서별 과제의 최소 점수는 할당된 반영 비율의 10%로 설정됩니다. 예를 들어, 특정 과제가 8%의 비율로 설정된 경우 최소 점수는 0.8%입니다.',
 6, 1),

-- 2023년 상반기 개인 과제 평가 정책
(2, '2023-06-01', '2023-07-30', 2023, '1st', 0.3,
 4, '2023-05-01', '2023-05-31',
 '개인평가 최종 환산 점수: 2023년 상반기 개인평가는 사원의 최종 평가 점수에 30% 비율로 반영됩니다. 과제 반영 비율 제한: 개인평가의 총 비율은 30%로 설정되어 있으며, 각 과제별로 10% 단위로 비율이 부여될 수 있습니다. 즉, 각 과제는 3%, 6%, 9%, ... 30% 범위 내에서 비율이 설정되며, 총 비율인 30%를 초과할 수 없습니다. 과제별 최소 점수: 모든 과제의 최소 점수는 할당된 반영 비율의 10%로 설정됩니다. 예를 들어, 특정 과제가 6%의 비율로 설정된 경우 최소 점수는 0.6%입니다.',
 6, 2),

-- 2023년 상반기 공통 과제 평가 정책
(3, '2023-06-01', '2023-07-30', 2023, '1st', 0.3,
 4, '2023-05-01', '2023-09-01',
 '공통평가 최종 환산 점수: 2023년 상반기 공통평가는 사원의 최종 평가 점수에 30%의 비율로 반영됩니다. 과제 반영 비율 제한: 공통평가의 총 비율은 30%로 설정되어 있으며, 각 과제별로 10% 단위로 비율이 부여될 수 있습니다. 즉, 각 과제는 3%, 6%, 9%, ... 30% 범위 내에서 비율이 설정되며, 총 비율인 30%를 초과할 수 없습니다. 과제별 최소 점수: 공통 과제의 최소 점수는 할당된 반영 비율의 10%로 설정됩니다. 예를 들어, 특정 과제가 9%의 비율로 설정된 경우 최소 점수는 0.9%입니다.',
 6, 3);


-- 등급

-- S 등급
INSERT INTO grade (grade_id, grade_name, start_ratio, end_ratio, absolute_grade_ratio, evaluation_policy_id)
VALUES
    (1, 'S', 0.00, 0.05, 0.05, 1),
    (2, 'S', 0.00, 0.05, 0.05, 2),
    (3, 'S', 0.00, 0.05, 0.05, 3),
-- A 등급
    (7, 'A', 0.05, 0.15, 0.15, 1),
    (8, 'A', 0.05, 0.15, 0.15, 2),
    (9, 'A', 0.05, 0.15, 0.15, 3),

-- B 등급
    (13, 'B', 0.15, 0.7, 0.7, 1),
    (14, 'B', 0.15, 0.7, 0.7, 2),
    (15, 'B', 0.15, 0.7, 0.7, 3),

-- C 등급
    (19, 'C', 0.7, 0.9, 0.9, 1),
    (20, 'C', 0.7, 0.9, 0.9, 2),
    (21, 'C', 0.7, 0.9, 0.9, 3),

-- D 등급
    (25, 'D', 0.9, 1.0, 1.0, 1),
    (26, 'D', 0.9, 1.0, 1.0, 2),
    (27, 'D', 0.9, 1.0, 1.0, 3);

-- 평가 테이블

INSERT INTO evaluation (evaluation_id, evaluation_type, fin_score, fin_grade, year, half, created_at, employee_id, evaluator_id)
VALUES
    (1, '자기평가', 100, 'S', 2023, '1st', '2023-05-15', 7, 7),
    (2, '리더평가', 96.04, 'S', 2023, '1st', '2023-05-15', 7, 8);

-- 과제 항목

INSERT INTO task_item (task_item_id, task_name, task_content, assigned_employee_count, task_type_id, employee_id, department_code, evaluation_policy_id, is_manager_written)
VALUES
    (1, 'SAP ERP 정기 패치 적용', '분기별 SAP ERP 시스템 보안 패치 및 기능 업데이트 진행', 1, 1, 8, 'DP006', 1, TRUE),
    (2, '랜섬웨어 대응 체계 구축', '랜섬웨어 대응 솔루션 도입 및 백업 시스템 이중화', 1, 1, 8, 'DP006', 1, TRUE),
    (3, '정보보안 관리체계 인증', 'ISMS-P 인증 획득을 위한 보안 체계 점검 및 개선', 1, 1, 8, 'DP006', 1, TRUE),
    (4, '정보처리기사 자격증 취득', '정보처리기사 자격증 취득을 위한 시험 준비 및 합격 달성', 1, 2, 7, 'DP006', 2, FALSE),
    (5, 'SQLD 자격증 취득', '데이터베이스 기초 지식 습득 및 SQLD 자격증 시험 합격', 1, 2, 7, 'DP006', 2, FALSE),
    (6, '오픽 AL 달성', 'OPIc 시험 준비 및 Advanced Low (AL) 등급 달성', 1, 2, 7, 'DP006', 2, FALSE),
    (7, '사내 캡스톤 대회 수상', '사내 캡스톤 프로젝트 대회 참가 및 입상 성과 달성', 1, 2, 7, 'DP006', 2, FALSE),
    (8, '데이터 분석 프로젝트 수행', '사내 데이터 분석 프로젝트 참여 및 성공적 수행', 1, 2, 7, 'DP006', 2, FALSE),
    (9, '매출액 5000억원 달성', '사업부 목표 매출액 5000억원 달성', 1, 3, 6, 'DP002', 3, FALSE),
    (10, '고객 만족도 90% 이상 달성', '고객 만족도 설문조사에서 90% 이상 달성', 1, 3, 6, 'DP002', 3, FALSE);


-- 과제항목별평가 테이블

INSERT INTO task_eval (
    task_eval_id,
    task_eval_name,
    task_eval_content,
    score,
    set_ratio,
    task_grade,
    performance_input,
    created_at,
    rel_eval_status,
    evaluation_id,
    modifiable_date,
    task_type_id,
    task_item_id
) VALUES

(1, 'SAP ERP 패치 적용 평가 - 한상민', '전체 패치 프로세스 관리', 95.5, 0.4, 'S', 'ERP 시스템 패치 적용 완료 및 안정화 100% 달성', '2023-06-01', false, 2, '2023-09-30', 1, 1),
(2, '정보보안 관리체계 인증 평가 - 한상민', '보안 정책 수립 및 구현', 87.5, 0.3, 'A', 'ISMS-P 인증 심사 준비 완료', '2023-06-01', false, 2, '2023-09-30', 1, 3),
(3, '랜섬웨어 대응 평가 - 한상민', '대응체계 구축', 85.5, 0.3, 'B', '랜섬웨어 대응 체계 90% 구축', '2023-06-01', false, 2, '2023-09-30', 1, 2),
(4, '정보처리기사 자격증 취득 평가', '정보처리기사 자격증 취득을 위한 준비 및 합격', 95.0, 0.2, 'S', '정보처리기사 자격증 취득 성공', '2023-06-01', false, 2, '2023-09-30', 2, 4),
(5, 'SQLD 자격증 취득 평가', 'SQLD 자격증을 위한 준비 및 시험 합격', 96.0, 0.2, 'S', 'SQLD 자격증 취득 성공', '2023-06-01', false, 2, '2023-09-30', 2, 5),
(6, '오픽 AL 등급 달성 평가', 'OPIc AL 등급 달성을 위한 준비 및 성취', 97.0, 0.2, 'S', 'OPIc AL 등급 달성', '2023-06-01', false, 2, '2023-09-30', 2, 6),
(7, '사내 캡스톤 대회 수상 평가', '캡스톤 프로젝트 대회 참가 및 입상', 98.0, 0.2, 'S', '사내 캡스톤 대회에서 입상', '2023-06-01', false, 2, '2023-09-30', 2, 7),
(8, '데이터 분석 프로젝트 수행 평가', '데이터 분석 프로젝트 참여 및 성공적 수행', 99.0, 0.2, 'S', '데이터 분석 프로젝트 성공적 수행', '2023-06-01', false, 2, '2023-09-30', 2, 8),
(9, '매출액 5000억원 달성 평가 - 한상민', '사업부 목표 매출액 5000억원 달성 여부 평가', 100, 0.5, 'S', '사업부 목표 매출액 5000억원을 달성하여 기업 성과에 기여', '2023-06-01', false, 2, '2023-09-30', 3, 9),
(10, '고객 만족도 90% 이상 달성 평가 - 한상민', '고객 만족도 90% 이상 달성 여부 평가', 100, 0.5, 'S', '고객 만족도 설문조사 결과 90% 이상의 만족도를 달성하여 긍정적인 기업 이미지를 구축', '2023-06-01', false, 2, '2023-09-30', 3, 10),

-- 리더평가와 동일한 과제항목에 대한 자기평가 점수
(11, 'SAP ERP 패치 적용 평가 - 한상민 (자기평가)', '전체 패치 프로세스 관리', 100, 0.4, 'S', 'ERP 시스템 패치 적용 완료 및 안정화 100% 달성', '2023-06-01', false, 1, '2023-09-30', 1, 1),
(12, '정보보안 관리체계 인증 평가 - 한상민 (자기평가)', '보안 정책 수립 및 구현', 100, 0.3, 'S', 'ISMS-P 인증 심사 준비 완료', '2023-06-01', false, 1, '2023-09-30', 1, 10),
(13, '랜섬웨어 대응 평가 - 한상민 (자기평가)', '대응체계 구축', 100, 0.3, 'S', '랜섬웨어 대응 체계 90% 구축', '2023-06-01', false, 1, '2023-09-30', 1, 9),
(14, '정보처리기사 자격증 취득 평가 (자기평가)', '정보처리기사 자격증 취득을 위한 준비 및 합격', 100, 0.2, 'S', '정보처리기사 자격증 취득 성공', '2023-06-01', false, 1, '2023-09-30', 2, 4),
(15, 'SQLD 자격증 취득 평가 (자기평가)', 'SQLD 자격증을 위한 준비 및 시험 합격', 100, 0.2, 'S', 'SQLD 자격증 취득 성공', '2023-06-01', false, 1, '2023-09-30', 2, 5),
(16, '오픽 AL 등급 달성 평가 (자기평가)', 'OPIc AL 등급 달성을 위한 준비 및 성취', 100, 0.2, 'S', 'OPIc AL 등급 달성', '2023-06-01', false, 1, '2023-09-30', 2, 6),
(17, '사내 캡스톤 대회 수상 평가 (자기평가)', '캡스톤 프로젝트 대회 참가 및 입상', 100, 0.2, 'S', '사내 캡스톤 대회에서 입상', '2023-06-01', false, 1, '2023-09-30', 2, 7),
(18, '데이터 분석 프로젝트 수행 평가 (자기평가)', '데이터 분석 프로젝트 참여 및 성공적 수행', 100, 0.2, 'S', '데이터 분석 프로젝트 성공적 수행', '2023-06-01', false, 1, '2023-09-30', 2, 8),
(19, '매출액 5000억원 달성 평가 - 한상민 (자기평가)', '사업부 목표 매출액 5000억원 달성 여부 평가', 100, 0.5, 'S', '사업부 목표 매출액 5000억원을 달성하여 기업 성과에 기여', '2023-06-01', false, 1, '2023-09-30', 3, 9),
(20, '고객 만족도 90% 이상 달성 평가 - 한상민 (자기평가)', '고객 만족도 90% 이상 달성 여부 평가', 100, 0.5, 'S', '고객 만족도 설문조사 결과 90% 이상의 만족도를 달성하여 긍정적인 기업 이미지를 구축', '2023-06-01', false, 1, '2023-09-30', 3, 10);


-- 평가정책별평가 테이블

INSERT INTO task_type_eval (
    task_type_eval_id,
    task_type_total_score,
    created_at,
    evaluation_id,
    evaluation_policy_id
) VALUES
-- 부서별 과제 평가 총 점수
(1, 36.04, '2023-07-01', 2, 1),

-- 개인 과제 평가 총 점수
(2, 29.1, '2023-07-01', 2, 2),

-- 공통 과제 평가 총 점수
(3, 30, '2023-07-01', 2, 3),

(4, 40.0, '2023-07-01', 1, 1),

-- 개인 과제 평가 총 점수 (자기평가)
(5, 30.0, '2023-07-01', 1, 2),

-- 공통 과제 평가 총 점수 (자기평가)
(6, 30.0, '2023-07-01', 1, 3);

-- 피드백 테이블

INSERT INTO feedback (
    feedback_id,
    content,
    created_at,
    evaluation_id
) VALUES
(1, '한상민 사원님, 이번 평가에서 우수한 성적을 거두신 것을 진심으로 축하드립니다! 높은 책임감과 성실함으로 프로젝트를 성공적으로 이끌어주셔서 감사드립니다. 특히 어려운 상황에서도 포기하지 않고 끝까지 노력하는 모습이 매우 인상 깊었습니다. 앞으로도 팀의 든든한 버팀목이 되어주시리라 믿습니다. 더 큰 성과를 이뤄나가시기를 응원합니다!', '2023-07-01', 2);
